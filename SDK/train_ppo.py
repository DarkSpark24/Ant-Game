from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SDK.training import (  # noqa: E402
    AntWarParallelEnv,
    PPOSelfPlayTrainer,
    PPOTrainerConfig,
    TrainingLogger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO self-play agent with PyTorch.")
    parser.add_argument("--batches", type=int, default=1, help="Number of PPO updates.")
    parser.add_argument("--episodes", type=int, default=2, help="Self-play episodes per update.")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="Gradient epochs per update batch.")
    parser.add_argument("--minibatch-size", type=int, default=256, help="Minibatch size for PPO updates.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clipping ratio.")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value loss coefficient.")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping norm.")
    parser.add_argument("--max-rounds", type=int, default=128, help="Hard cap for each self-play match.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--max-actions", type=int, default=96, help="Candidate action budget.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="First hidden layer width.")
    parser.add_argument("--hidden-dim2", type=int, default=128, help="Second hidden layer width.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ppo_latest.pt", help="Path for latest checkpoint.")
    parser.add_argument("--resume-from", type=str, default=None, help="Optional checkpoint path for warm start.")
    parser.add_argument("--evaluation-episodes", type=int, default=2, help="Evaluation episodes after each update.")
    parser.add_argument("--device", type=str, default=None, help="Training device override, e.g. cpu/cuda/cuda:0.")
    parser.add_argument("--log-dir", type=str, default="logs/train_ppo", help="Base directory for training logs.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name under log directory.")
    parser.add_argument(
        "--prefer-native-backend",
        action="store_true",
        help="Prefer the optional native backend for environment resets if it is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PPOTrainerConfig(
        batches=args.batches,
        episodes=args.episodes,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        learning_rate=args.learning_rate,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        max_rounds=args.max_rounds,
        max_actions=args.max_actions,
        hidden_dim=args.hidden_dim,
        hidden_dim2=args.hidden_dim2,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume_from,
        evaluation_episodes=args.evaluation_episodes,
        device=args.device,
    )
    logger = TrainingLogger(base_dir=args.log_dir, run_name=args.run_name)
    logger.log_config(
        {
            "argv": vars(args),
            "trainer_config": asdict(config),
        }
    )
    try:
        trainer = PPOSelfPlayTrainer(
            env_factory=lambda seed=0: AntWarParallelEnv(
                seed=seed,
                max_actions=args.max_actions,
                prefer_native_backend=args.prefer_native_backend,
            ),
            config=config,
            logger=logger,
        )
        history, samples = trainer.train()
        latest = history[-1] if history else {}
        result = {
            "episodes": args.episodes,
            "batches": args.batches,
            "max_rounds": args.max_rounds,
            "checkpoint": str(Path(args.checkpoint)),
            "log_dir": str(logger.run_dir),
            "resume_from": args.resume_from,
            "training_entrypoint": "SDK/train_ppo.py",
            "trainer_logic_hook": "PPOSelfPlayTrainer.update_from_batch()",
            "agent_logic_file": "AI/ai_example.py",
            "policy_backend": "SDK.training.ppo_torch.ActorCriticNet",
            "latest_metrics": latest,
            "history": history,
            "samples": [asdict(summary) for summary in samples[: min(len(samples), 3)]],
        }
        logger.log_summary(result)
        print(json.dumps(result, indent=2, sort_keys=True))
    except Exception as exc:
        logger.log_error(f"training failed: {exc}")
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
