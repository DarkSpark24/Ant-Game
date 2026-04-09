from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from AI.ai_example import AI as ExampleAI
from SDK.backend import GameState


def test_ai_example_can_choose_operations() -> None:
    agent = ExampleAI(seed=5, max_actions=16)
    state = GameState.initial(seed=5)
    operations = agent.choose_operations(state, 0)
    assert isinstance(operations, list)


def test_train_example_script_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "SDK/train_example.py", "--seed", "2", "--max-actions", "16"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "AI/ai_example.py" in completed.stdout
    assert "SDK.backend" in completed.stdout


def test_train_example_shell_runs() -> None:
    completed = subprocess.run(
        ["bash", "SDK/train_example.sh", "--seed", "2", "--max-actions", "16"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "training_entrypoint" in completed.stdout


def test_train_mcts_script_runs_short_scaffold() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "SDK/train_mcts.py",
            "--episodes",
            "1",
            "--iterations",
            "2",
            "--max-depth",
            "1",
            "--max-rounds",
            "2",
            "--seed",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"episodes": 1' in completed.stdout
    assert "update_from_episodes" in completed.stdout


def test_train_mcts_shell_runs_short_scaffold() -> None:
    completed = subprocess.run(
        [
            "bash",
            "SDK/train_mcts.sh",
            "--episodes",
            "1",
            "--iterations",
            "2",
            "--max-depth",
            "1",
            "--max-rounds",
            "2",
            "--seed",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"episodes": 1' in completed.stdout


def test_train_mcts_script_writes_logs(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    completed = subprocess.run(
        [
            sys.executable,
            "SDK/train_mcts.py",
            "--episodes",
            "1",
            "--batches",
            "1",
            "--iterations",
            "2",
            "--max-depth",
            "1",
            "--max-rounds",
            "2",
            "--seed",
            "5",
            "--evaluation-episodes",
            "1",
            "--log-dir",
            str(log_root),
            "--run-name",
            "smoke",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    run_dir = Path(payload["log_dir"])
    assert run_dir == log_root / "smoke"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "train.log").exists()


def test_train_ppo_script_runs_short_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    log_root = tmp_path / "logs"
    completed = subprocess.run(
        [
            sys.executable,
            "SDK/train_ppo.py",
            "--episodes",
            "1",
            "--batches",
            "1",
            "--ppo-epochs",
            "1",
            "--minibatch-size",
            "32",
            "--max-rounds",
            "4",
            "--seed",
            "7",
            "--evaluation-episodes",
            "1",
            "--log-dir",
            str(log_root),
            "--run-name",
            "smoke",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    run_dir = Path(payload["log_dir"])
    assert run_dir == log_root / "smoke"
    assert (run_dir / "config.json").exists()
    assert (run_dir / "events.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "train.log").exists()
    assert (run_dir / "metrics.csv").exists()
