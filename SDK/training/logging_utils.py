from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import csv
import json
import logging
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    return value


class TrainingLogger:
    def __init__(self, base_dir: str | Path, run_name: str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        stem = run_name or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.run_dir = self._allocate_run_dir(stem)
        self.text_log_path = self.run_dir / "train.log"
        self.events_path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.config_path = self.run_dir / "config.json"
        self.metrics_csv_path = self.run_dir / "metrics.csv"
        self.curves_png_path = self.run_dir / "curves.png"
        self._events_fp = self.events_path.open("a", encoding="utf-8")
        self._batch_metrics: list[dict[str, Any]] = []
        self._plot_warning_emitted = False
        self.logger = self._build_logger()

    def _allocate_run_dir(self, stem: str) -> Path:
        candidate = self.base_dir / stem
        suffix = 1
        while candidate.exists() and any(candidate.iterdir()):
            candidate = self.base_dir / f"{stem}-{suffix:02d}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _build_logger(self) -> logging.Logger:
        logger_name = f"agent_tradition.training.{self.run_dir.as_posix()}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.text_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        return logger

    def _write_json(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")

    def log_event(self, kind: str, payload: dict[str, Any]) -> None:
        record = {
            "kind": kind,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self._events_fp.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
        self._events_fp.flush()

    def log_config(self, payload: dict[str, Any]) -> None:
        self._write_json(self.config_path, payload)
        self.log_event("config", payload)
        self.logger.info("training run initialized run_dir=%s", self.run_dir)

    def log_episode(self, batch_index: int, episode_index: int, payload: dict[str, Any]) -> None:
        self.log_event(
            "episode",
            {
                "batch_index": batch_index,
                "episode_index": episode_index,
                **payload,
            },
        )
        self.logger.info(
            "episode batch=%s episode=%s rounds=%s winner=%s reward_p0=%.4f reward_p1=%.4f",
            batch_index,
            episode_index,
            payload.get("rounds"),
            payload.get("winner"),
            float(payload.get("reward_player_0", 0.0)),
            float(payload.get("reward_player_1", 0.0)),
        )

    def log_batch_metrics(self, batch_index: int, payload: dict[str, Any]) -> None:
        row = {
            "batch_index": batch_index,
            **payload,
        }
        self._batch_metrics.append(row)
        self._write_metrics_csv()
        self._write_curves_png()
        self.log_event(
            "batch_metrics",
            row,
        )
        self.logger.info(
            "batch=%s policy_loss=%.4f value_loss=%.4f entropy=%.4f eval_win_rate=%.4f eval_return=%.4f mean_reward=%.4f samples=%s",
            batch_index,
            float(payload.get("policy_loss", 0.0)),
            float(payload.get("value_loss", 0.0)),
            float(payload.get("entropy", 0.0)),
            float(payload.get("eval_win_rate", 0.0)),
            float(payload.get("eval_return", 0.0)),
            float(payload.get("mean_reward", 0.0)),
            payload.get("samples"),
        )

    def _write_metrics_csv(self) -> None:
        if not self._batch_metrics:
            return
        fieldnames = sorted({key for row in self._batch_metrics for key in row.keys()})
        with self.metrics_csv_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._batch_metrics:
                writer.writerow(row)

    def _write_curves_png(self) -> None:
        if len(self._batch_metrics) < 1:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            if not self._plot_warning_emitted:
                self._plot_warning_emitted = True
                message = f"matplotlib unavailable, skip curves.png generation: {exc}"
                self.logger.warning("%s", message)
                self.log_event("warning", {"message": message})
            return

        x_values = [float(row.get("batch_index", idx)) for idx, row in enumerate(self._batch_metrics)]

        def _series(name: str) -> list[float | None]:
            values: list[float | None] = []
            for row in self._batch_metrics:
                if name not in row:
                    values.append(None)
                    continue
                try:
                    values.append(float(row[name]))
                except (TypeError, ValueError):
                    values.append(None)
            return values

        tracked = [
            ("mean_reward", "mean_reward"),
            ("eval_return", "eval_return"),
            ("policy_loss", "policy_loss"),
            ("value_loss", "value_loss"),
            ("entropy", "entropy"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.reshape(-1)
        panels = [
            ("Reward", ["mean_reward", "eval_return"]),
            ("Policy", ["policy_loss"]),
            ("Value", ["value_loss"]),
            ("Entropy", ["entropy"]),
        ]
        for axis, (title, names) in zip(axes, panels):
            for name in names:
                values = _series(name)
                pairs = [(x, y) for x, y in zip(x_values, values) if y is not None]
                if not pairs:
                    continue
                xs, ys = zip(*pairs)
                label = dict(tracked).get(name, name)
                axis.plot(xs, ys, marker="o", label=label)
            axis.set_title(title)
            axis.set_xlabel("batch")
            axis.grid(True, alpha=0.35)
            if axis.lines:
                axis.legend()
        fig.tight_layout()
        fig.savefig(self.curves_png_path, dpi=140)
        plt.close(fig)

    def log_checkpoint(self, batch_index: int, checkpoint_path: str | Path) -> None:
        payload = {
            "batch_index": batch_index,
            "checkpoint_path": str(checkpoint_path),
        }
        self.log_event("checkpoint", payload)
        self.logger.info("checkpoint batch=%s path=%s", batch_index, checkpoint_path)

    def log_summary(self, payload: dict[str, Any]) -> None:
        self._write_json(self.summary_path, payload)
        self.log_event("summary", payload)
        self.logger.info("training completed")

    def log_error(self, message: str) -> None:
        self.log_event("error", {"message": message})
        self.logger.error("%s", message)

    def close(self) -> None:
        self._events_fp.close()
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
