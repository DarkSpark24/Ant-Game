import importlib

import numpy as np
import pytest


try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


def _blank_obs(current_player: int = 0):
    from logic.constant import row, col

    return {
        "board_owner": np.full((row, col), -1, dtype=np.int8),
        "board_army": np.zeros((row, col), dtype=np.int32),
        "board_terrain": np.zeros((row, col), dtype=np.int8),
        "generals_owner": np.full((row, col), -1, dtype=np.int8),
        "coins": np.array([50, 50], dtype=np.int32),
        "rest_moves": np.array([2, 2], dtype=np.int32),
        "current_player": int(current_player),
    }


def test_selfplay_ai_uses_sdk_observation_encoding_contract(plain_state):
    from AI import ai_selfplay
    from SDK.policies import obs_to_tensor
    from tests.conftest import place_general

    s = plain_state
    place_general(s, "main", 0, 0, 0)
    place_general(s, "sub", 1, 1, 1)
    s.board[0][0].army = 20001
    s.board[1][1].army = 7

    obs = ai_selfplay._build_obs(s, 0)
    assert obs["board_army"][0][0] == ai_selfplay.MAX_ARMY_OBS
    np.testing.assert_allclose(ai_selfplay._obs_to_tensor(obs), obs_to_tensor(obs))


def test_selfplay_ai_falls_back_to_end_turn_without_model(tmp_path, plain_state, monkeypatch):
    from AI import ai_selfplay

    monkeypatch.setattr(ai_selfplay, "MODEL_PATH", str(tmp_path / "missing_model.pt"))
    monkeypatch.delattr(ai_selfplay, "_POLICY", raising=False)

    assert ai_selfplay.policy(1, 0, plain_state) == [[8]]


def test_training_entrypoints_fail_cleanly_without_torch(tmp_path):
    train_selfplay = importlib.import_module("SDK.train_selfplay")

    if train_selfplay.torch is not None:
        pytest.skip("PyTorch is installed; torch-backed training behavior is covered separately.")

    with pytest.raises(RuntimeError, match="PyTorch not installed"):
        train_selfplay.train_legal(1, str(tmp_path / "legal.pt"))
    with pytest.raises(RuntimeError, match="PyTorch not installed"):
        train_selfplay.train_win(1, str(tmp_path / "win.pt"))


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_logits_to_action_falls_back_to_end_turn_without_general():
    from SDK.policies import logits_to_action
    from logic.constant import row, col

    obs = _blank_obs(current_player=0)
    logits = torch.full((9 + row + col + 4 + 5 + 10,), -1000.0)
    logits[1] = 1000.0  # force ArmyMove

    action, logp, is_direct_end = logits_to_action(logits, obs)

    assert action.shape == (8,)
    assert int(action[0]) == 0
    assert bool(is_direct_end) is True
    assert torch.isfinite(logp)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_sdk_training_policy_weights_load_into_selfplay_ai():
    from AI.ai_selfplay import SmallPolicy as AiPolicy
    from SDK.policies import PolicyConfig, SmallPolicy as SdkPolicy

    sdk_policy = SdkPolicy(PolicyConfig())
    ai_policy = AiPolicy()

    ai_policy.load_state_dict(sdk_policy.state_dict())

    sdk_state = sdk_policy.state_dict()
    ai_state = ai_policy.state_dict()
    assert list(sdk_state.keys()) == list(ai_state.keys())
    for key in sdk_state:
        assert sdk_state[key].shape == ai_state[key].shape


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_train_legal_zero_episode_saves_valid_policy_state(tmp_path, monkeypatch):
    from SDK import train_selfplay
    from SDK.policies import PolicyConfig, SmallPolicy

    monkeypatch.setattr(train_selfplay, "make_env", lambda render_mode=None: object())
    monkeypatch.setattr(train_selfplay.tqdm, "tqdm", lambda xs: xs)

    save_path = tmp_path / "legal.pt"
    train_selfplay.train_legal(0, str(save_path), seed=7)

    saved = torch.load(save_path, map_location="cpu")
    expected = SmallPolicy(PolicyConfig()).state_dict()
    assert list(saved.keys()) == list(expected.keys())
    for key in expected:
        assert saved[key].shape == expected[key].shape


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_train_win_zero_episode_preserves_loaded_warm_start(tmp_path, monkeypatch):
    from SDK import train_selfplay
    from SDK.policies import PolicyConfig, SmallPolicy

    monkeypatch.setattr(train_selfplay, "make_env", lambda render_mode=None: object())
    monkeypatch.setattr(train_selfplay.tqdm, "tqdm", lambda xs: xs)

    warm_policy = SmallPolicy(PolicyConfig())
    with torch.no_grad():
        for idx, param in enumerate(warm_policy.parameters(), start=1):
            param.fill_(idx / 100.0)

    warm_path = tmp_path / "warm.pt"
    save_path = tmp_path / "win.pt"
    torch.save(warm_policy.state_dict(), warm_path)

    train_selfplay.train_win(0, str(save_path), seed=11, legal_model_path=str(warm_path))

    saved = torch.load(save_path, map_location="cpu")
    expected = warm_policy.state_dict()
    assert list(saved.keys()) == list(expected.keys())
    for key in expected:
        assert torch.equal(saved[key], expected[key])
