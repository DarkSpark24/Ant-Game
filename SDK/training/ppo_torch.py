from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np

from SDK.training.env import AntWarParallelEnv
from SDK.training.logging_utils import TrainingLogger
from SDK.utils.features import FeatureExtractor

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - guarded by runtime checks in train entrypoint/tests.
    torch = None
    nn = None


@dataclass(slots=True)
class PPOTrainerConfig:
    batches: int = 1
    episodes: int = 4
    ppo_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    max_rounds: int = 128
    max_actions: int = 96
    hidden_dim: int = 256
    hidden_dim2: int = 128
    seed: int = 0
    checkpoint_path: str = "checkpoints/ppo_latest.pt"
    resume_from: str | None = None
    evaluation_episodes: int = 2
    device: str | None = None


@dataclass(slots=True)
class PPOStep:
    observation: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


@dataclass(slots=True)
class PPOEpisodeSummary:
    seed: int
    rounds: int
    winner: int | None
    reward_player_0: float
    reward_player_1: float


@dataclass(slots=True)
class PPORolloutBatch:
    observations: np.ndarray
    masks: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray


class ActorCriticNet(nn.Module if nn is not None else object):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, hidden_dim2: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim2, action_dim)
        self.value_head = nn.Linear(hidden_dim2, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(observation)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value


class PPOSelfPlayTrainer:
    def __init__(
        self,
        env_factory,
        config: PPOTrainerConfig | None = None,
        logger: TrainingLogger | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PPO trainer requires PyTorch. Please install torch first.")
        self.env_factory = env_factory
        self.config = config or PPOTrainerConfig()
        self.logger = logger
        self.feature_extractor = FeatureExtractor(max_actions=self.config.max_actions)
        self.device = self._resolve_device(self.config.device)
        self._set_global_seeds(self.config.seed)

        warmup_env = self.env_factory(seed=self.config.seed)
        observations, _ = warmup_env.reset(seed=self.config.seed)
        sample = observations["player_0"]
        obs_dim = len(self.feature_extractor.flatten_observation(sample))
        action_dim = len(sample["action_mask"])
        warmup_env.close()

        self.model = ActorCriticNet(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_dim2=self.config.hidden_dim2,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)

    def _resolve_device(self, override: str | None) -> torch.device:
        if override is not None:
            return torch.device(override)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _set_global_seeds(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    def _masked_logits(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        safe_mask = mask.clone()
        invalid_rows = safe_mask.sum(dim=1) <= 0
        if invalid_rows.any():
            safe_mask[invalid_rows, 0] = 1.0
        return logits.masked_fill(safe_mask <= 0, -1e9)

    def _policy_step(self, observation: np.ndarray, mask: np.ndarray, explore: bool = True) -> tuple[int, float, float]:
        obs_tensor = self._to_tensor(observation[None, :])
        mask_tensor = self._to_tensor(mask[None, :])
        with torch.no_grad():
            logits, value = self.model(obs_tensor)
            masked_logits = self._masked_logits(logits, mask_tensor)
            dist = torch.distributions.Categorical(logits=masked_logits)
            if explore:
                action = int(dist.sample().item())
            else:
                action = int(torch.argmax(masked_logits, dim=1).item())
            log_prob = float(dist.log_prob(torch.tensor([action], device=self.device)).item())
            value_scalar = float(value.item())
        return action, log_prob, value_scalar

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        bootstrap_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros(len(rewards), dtype=np.float32)
        gae = 0.0
        for index in reversed(range(len(rewards))):
            next_value = bootstrap_value if index == len(rewards) - 1 else values[index + 1]
            mask = 0.0 if dones[index] else 1.0
            delta = rewards[index] + self.config.gamma * next_value * mask - values[index]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[index] = gae
        returns = advantages + np.asarray(values, dtype=np.float32)
        return returns, advantages

    def collect_episode(self, seed: int, explore: bool = True) -> tuple[PPORolloutBatch, PPOEpisodeSummary]:
        env: AntWarParallelEnv = self.env_factory(seed=seed)
        try:
            observations, _ = env.reset(seed=seed)
            traces: dict[str, list[PPOStep]] = {agent: [] for agent in env.possible_agents}
            total_reward = {agent: 0.0 for agent in env.possible_agents}
            rounds = 0

            while env.agents and rounds < self.config.max_rounds:
                actions: dict[str, int] = {}
                for agent in env.possible_agents:
                    current = observations[agent]
                    flat = self.feature_extractor.flatten_observation(current).astype(np.float32, copy=False)
                    mask = current["action_mask"].astype(np.float32, copy=False)
                    action, log_prob, value = self._policy_step(flat, mask, explore=explore)
                    traces[agent].append(
                        PPOStep(
                            observation=flat,
                            mask=mask,
                            action=action,
                            log_prob=log_prob,
                            value=value,
                            reward=0.0,
                            done=False,
                        )
                    )
                    actions[agent] = action

                observations, rewards, terminations, truncations, _ = env.step(actions)
                rounds += 1
                done = all(terminations.values()) or all(truncations.values()) or rounds >= self.config.max_rounds
                for agent in env.possible_agents:
                    total_reward[agent] += float(rewards[agent])
                    traces[agent][-1].reward = float(rewards[agent])
                    traces[agent][-1].done = bool(done)
                if done:
                    break

            rows_obs = []
            rows_mask = []
            rows_action = []
            rows_log_prob = []
            rows_return = []
            rows_advantage = []
            for agent in env.possible_agents:
                steps = traces[agent]
                rewards = [step.reward for step in steps]
                values = [step.value for step in steps]
                dones = [step.done for step in steps]
                returns, advantages = self._compute_gae(
                    rewards=rewards,
                    values=values,
                    dones=dones,
                    bootstrap_value=0.0,
                )
                for idx, step in enumerate(steps):
                    rows_obs.append(step.observation)
                    rows_mask.append(step.mask)
                    rows_action.append(step.action)
                    rows_log_prob.append(step.log_prob)
                    rows_return.append(float(returns[idx]))
                    rows_advantage.append(float(advantages[idx]))

            batch = PPORolloutBatch(
                observations=np.asarray(rows_obs, dtype=np.float32),
                masks=np.asarray(rows_mask, dtype=np.float32),
                actions=np.asarray(rows_action, dtype=np.int64),
                old_log_probs=np.asarray(rows_log_prob, dtype=np.float32),
                returns=np.asarray(rows_return, dtype=np.float32),
                advantages=np.asarray(rows_advantage, dtype=np.float32),
            )
            summary = PPOEpisodeSummary(
                seed=seed,
                rounds=rounds,
                winner=env.state.winner,
                reward_player_0=round(total_reward["player_0"], 4),
                reward_player_1=round(total_reward["player_1"], 4),
            )
            return batch, summary
        finally:
            env.close()

    def _merge_batches(self, batches: list[PPORolloutBatch]) -> PPORolloutBatch:
        return PPORolloutBatch(
            observations=np.concatenate([batch.observations for batch in batches], axis=0),
            masks=np.concatenate([batch.masks for batch in batches], axis=0),
            actions=np.concatenate([batch.actions for batch in batches], axis=0),
            old_log_probs=np.concatenate([batch.old_log_probs for batch in batches], axis=0),
            returns=np.concatenate([batch.returns for batch in batches], axis=0),
            advantages=np.concatenate([batch.advantages for batch in batches], axis=0),
        )

    def update_from_batch(self, batch: PPORolloutBatch) -> dict[str, float]:
        observations = self._to_tensor(batch.observations)
        masks = self._to_tensor(batch.masks)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        old_log_probs = self._to_tensor(batch.old_log_probs)
        returns = self._to_tensor(batch.returns)
        advantages = self._to_tensor(batch.advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        batch_size = int(actions.shape[0])
        indices = np.arange(batch_size)
        minibatch_size = max(1, min(self.config.minibatch_size, batch_size))
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        self.model.train()
        for _ in range(self.config.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                mb_idx_np = indices[start : start + minibatch_size]
                mb_idx = torch.as_tensor(mb_idx_np, dtype=torch.long, device=self.device)

                obs_mb = observations.index_select(0, mb_idx)
                mask_mb = masks.index_select(0, mb_idx)
                actions_mb = actions.index_select(0, mb_idx)
                old_log_probs_mb = old_log_probs.index_select(0, mb_idx)
                returns_mb = returns.index_select(0, mb_idx)
                advantages_mb = advantages.index_select(0, mb_idx)

                logits, values = self.model(obs_mb)
                masked_logits = self._masked_logits(logits, mask_mb)
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(actions_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs_mb)
                unclipped = ratio * advantages_mb
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages_mb
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = torch.mean((values - returns_mb) ** 2)
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        with torch.no_grad():
            logits, values = self.model(observations)
            masked_logits = self._masked_logits(logits, masks)
            dist = torch.distributions.Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions)
            approx_kl = torch.mean(old_log_probs - new_log_probs).item()
            clip_fraction = torch.mean((torch.abs(torch.exp(new_log_probs - old_log_probs) - 1.0) > self.config.clip_ratio).float()).item()

        return {
            "policy_loss": float(np.mean(policy_losses) if policy_losses else 0.0),
            "value_loss": float(np.mean(value_losses) if value_losses else 0.0),
            "entropy": float(np.mean(entropies) if entropies else 0.0),
            "approx_kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
            "mean_return": float(np.mean(batch.returns)),
            "mean_advantage": float(np.mean(batch.advantages)),
            "samples": float(batch_size),
            "device": str(self.device),
        }

    def _evaluate_episode(self, seed: int) -> float:
        env: AntWarParallelEnv = self.env_factory(seed=seed)
        try:
            observations, _ = env.reset(seed=seed)
            total_reward = 0.0
            rounds = 0
            while env.agents and rounds < self.config.max_rounds:
                actions = {}
                for agent in env.possible_agents:
                    current = observations[agent]
                    flat = self.feature_extractor.flatten_observation(current).astype(np.float32, copy=False)
                    mask = current["action_mask"].astype(np.float32, copy=False)
                    action, _, _ = self._policy_step(flat, mask, explore=False)
                    actions[agent] = action
                observations, rewards, terminations, truncations, _ = env.step(actions)
                total_reward += float(rewards["player_0"])
                rounds += 1
                if all(terminations.values()) or all(truncations.values()):
                    break
            return total_reward
        finally:
            env.close()

    def evaluate_policy(self, episodes: int | None = None) -> dict[str, float]:
        games = episodes if episodes is not None else self.config.evaluation_episodes
        if games <= 0:
            return {"eval_episodes": 0.0, "eval_return": 0.0}
        returns = [self._evaluate_episode(self.config.seed + 50_000 + i) for i in range(games)]
        return {
            "eval_episodes": float(games),
            "eval_return": float(np.mean(returns)),
        }

    def save_checkpoint(self) -> str:
        path = Path(self.config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(payload, path)
        return str(path)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        payload = torch.load(path, map_location=self.device)
        model_state = payload.get("model_state_dict", payload)
        self.model.load_state_dict(model_state)
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)

    def train(self, num_batches: int | None = None) -> tuple[list[dict[str, float]], list[PPOEpisodeSummary]]:
        updates = num_batches if num_batches is not None else self.config.batches
        history: list[dict[str, float]] = []
        summaries: list[PPOEpisodeSummary] = []
        for batch_index in range(updates):
            episode_batches: list[PPORolloutBatch] = []
            episode_summaries: list[PPOEpisodeSummary] = []
            for episode_offset in range(self.config.episodes):
                seed = self.config.seed + batch_index * 1_000 + episode_offset
                rollout, summary = self.collect_episode(seed=seed, explore=True)
                episode_batches.append(rollout)
                episode_summaries.append(summary)
                if self.logger is not None:
                    self.logger.log_episode(batch_index=batch_index, episode_index=episode_offset, payload=asdict(summary))

            merged = self._merge_batches(episode_batches)
            metrics = self.update_from_batch(merged)
            metrics["batch"] = float(batch_index)
            metrics["episodes"] = float(self.config.episodes)
            metrics["mean_reward"] = float(
                np.mean([summary.reward_player_0 for summary in episode_summaries])
            )
            metrics.update(self.evaluate_policy())
            checkpoint_path = self.save_checkpoint()
            metrics["checkpoint_saved"] = 1.0
            history.append(metrics)
            summaries.extend(episode_summaries)
            if self.logger is not None:
                self.logger.log_batch_metrics(batch_index=batch_index, payload=metrics)
                self.logger.log_checkpoint(batch_index=batch_index, checkpoint_path=checkpoint_path)
        return history, summaries
