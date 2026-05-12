import datetime
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .utils import save_json


WEIGHTED_VAL_ACCURACY_KEY = "weighted_val_accuracy"
DEFAULT_WEIGHTED_VAL_ACCURACY_WEIGHTS: Dict[int, float] = {
    1: 0.40,
    3: 0.25,
    5: 0.20,
    10: 0.15,
}


@dataclass(frozen=True)
class CheckpointPolicy:
    """Resolved checkpoint callback settings used to build Lightning callbacks."""

    monitor: Optional[str]
    mode: str
    filename: str
    save_top_k: int
    every_n_train_steps: Optional[int]


def configure_runtime(log_cfg: Dict[str, Any]) -> None:
    """Configure runtime precision and optional cuDNN benchmarking."""
    torch.set_float32_matmul_precision(log_cfg.get("matmul_precision", "high"))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = bool(log_cfg.get("cudnn_benchmark", True))


def resolve_artifact_dir(cfg: Dict[str, Any], stage: int) -> Path:
    """Return the artifact/checkpoint output directory for the selected stage."""
    checkpoint_cfg = cfg.get("checkpoint", {})
    if checkpoint_cfg.get("dirpath"):
        return Path(checkpoint_cfg["dirpath"])

    output_dir = cfg.get("training", {}).get("output_dir")
    if output_dir:
        return Path(output_dir) / f"checkpoints_stage_{stage}"

    return Path("artifacts") / f"checkpoints_stage_{stage}"


def _resolve_logger_dir(log_cfg: Dict[str, Any], artifact_dir: Path) -> Path:
    configured_log_dir = Path(log_cfg.get("log_dir", "logs"))
    if configured_log_dir.is_absolute():
        return configured_log_dir
    return artifact_dir / configured_log_dir


def build_logger(log_cfg: Dict[str, Any], stage: int, artifact_dir: Path) -> TensorBoardLogger:
    """Create the TensorBoard logger and ensure its output directory exists."""
    logger_root = _resolve_logger_dir(log_cfg, artifact_dir=artifact_dir)
    logger_root.mkdir(parents=True, exist_ok=True)
    return TensorBoardLogger(
        save_dir=str(logger_root),
        name=log_cfg.get("experiment_name", f"stage{stage}"),
    )


def save_config_snapshot(cfg: Dict[str, Any], artifact_dir: Path) -> None:
    """Persist a JSON snapshot of the active config for reproducibility."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_json(str(artifact_dir / "config.json"), cfg)


def save_tracked_code_snapshot(artifact_dir: Path) -> Optional[Path]:
    """Archive the full project root into a zip snapshot in *artifact_dir*."""
    root_path = Path(__file__).resolve().parents[1]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = artifact_dir / f"code_snapshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.zip"

    try:
        with zipfile.ZipFile(
            snapshot_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for fp in sorted(root_path.rglob("*")):
                if fp == snapshot_path or not fp.is_file():
                    continue
                rel = fp.relative_to(root_path)
                arcname = (Path(root_path.name) / rel).as_posix()
                archive.write(fp, arcname=arcname)
    except Exception:
        try:
            snapshot_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    return snapshot_path


def warn_if_stage_and_config_mismatch(cfg_path: str, stage: int) -> None:
    """Print a warning when config filename hints do not match selected stage."""
    cfg_name = Path(cfg_path).name.lower()
    if "stage1" in cfg_name and stage != 1:
        print(f"[warn] Config '{cfg_path}' looks like stage1 config but training.stage={stage}.")
    if "stage2" in cfg_name and stage != 2:
        print(f"[warn] Config '{cfg_path}' looks like stage2 config but training.stage={stage}.")


def _default_mode_for_monitor(monitor_name: str) -> str:
    """Infer checkpoint mode from monitor name (loss-like -> min, else max)."""
    lowered = monitor_name.lower()
    if "accuracy" in lowered or "_acc" in lowered or lowered.endswith("acc"):
        return "max"
    minimize_tokens = ("loss", "error", "perplexity", "nll")
    if any(token in lowered for token in minimize_tokens):
        return "min"
    return "max"


def _parse_weighted_val_accuracy_weights(raw: Any) -> Dict[int, float]:
    """Parse configured weighted-accuracy weights into ``{k: weight}``.

    Accepts either a mapping (``{1: 0.4, 3: 0.25, ...}`` or string keys
    ``{"top1": 0.4, "1": 0.4}``) or a list of ``{k, weight}`` entries.
    Falls back to :data:`DEFAULT_WEIGHTED_VAL_ACCURACY_WEIGHTS` when ``raw``
    is empty.
    """
    if raw is None:
        return dict(DEFAULT_WEIGHTED_VAL_ACCURACY_WEIGHTS)

    parsed: Dict[int, float] = {}

    def _coerce_k(key: Any) -> Optional[int]:
        if isinstance(key, bool):
            return None
        if isinstance(key, int):
            return int(key)
        if isinstance(key, str):
            stripped = key.strip().lower()
            if stripped.startswith("top"):
                stripped = stripped[3:]
            try:
                return int(stripped)
            except ValueError:
                return None
        return None

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            k = _coerce_k(key)
            if k is None:
                continue
            parsed[k] = float(value)
    elif isinstance(raw, (list, tuple)):
        for entry in raw:
            if isinstance(entry, Mapping):
                k = _coerce_k(entry.get("k", entry.get("top_k", entry.get("topk"))))
                if k is None:
                    continue
                parsed[k] = float(entry.get("weight", entry.get("w", 0.0)))
    if not parsed:
        return dict(DEFAULT_WEIGHTED_VAL_ACCURACY_WEIGHTS)
    return parsed


def _default_filename_for_monitor(monitor_name: str) -> str:
    """Build a safe default checkpoint filename template for the chosen monitor."""
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", monitor_name):
        return f"epoch{{epoch:02d}}-step{{step}}-{monitor_name}{{{monitor_name}:.4f}}"
    return "epoch{epoch:02d}-step{step}"


def _coerce_monitor_name(raw_monitor: Any) -> Optional[str]:
    """Normalize configured monitor value to a trimmed string or None."""
    monitor = str(raw_monitor).strip() if raw_monitor is not None else ""
    return monitor if monitor else None


def resolve_checkpoint_policy(
    cfg: Dict[str, Any],
    has_val: bool,
    val_every: Optional[int],
    every_steps: Optional[int],
) -> CheckpointPolicy:
    """Resolve effective checkpoint monitor, mode, naming, and save cadence."""
    ckpt_cfg = cfg.get("checkpoint", {})
    monitor = _coerce_monitor_name(ckpt_cfg.get("monitor"))

    if monitor is None:
        if has_val and (every_steps is None or val_every is not None):
            return CheckpointPolicy(
                monitor="val_loss",
                mode="min",
                filename="epoch{epoch:02d}-step{step}-val_loss{val_loss:.4f}",
                save_top_k=int(ckpt_cfg.get("save_top_k", 3)),
                every_n_train_steps=every_steps,
            )
        return CheckpointPolicy(
            monitor=None,
            mode="min",
            filename="epoch{epoch:02d}-step{step}",
            save_top_k=-1,
            every_n_train_steps=every_steps,
        )

    monitor_uses_val = monitor.startswith("val_") or monitor == WEIGHTED_VAL_ACCURACY_KEY
    if monitor_uses_val and not has_val:
        print(
            f"[checkpoint] warning: monitor='{monitor}' requested without a validation set; "
            "falling back to unmonitored checkpointing."
        )
        return CheckpointPolicy(
            monitor=None,
            mode="min",
            filename="epoch{epoch:02d}-step{step}",
            save_top_k=-1,
            every_n_train_steps=every_steps,
        )

    raw_mode = ckpt_cfg.get("mode")
    if raw_mode is None:
        raw_mode = ckpt_cfg.get("monitor_mode")
    mode = str(raw_mode or _default_mode_for_monitor(monitor)).lower()
    if mode not in {"min", "max"}:
        inferred = _default_mode_for_monitor(monitor)
        print(f"[checkpoint] warning: invalid mode='{mode}' for monitor='{monitor}'. Using mode='{inferred}'.")
        mode = inferred

    effective_every_steps = every_steps
    if (
        monitor_uses_val
        and val_every is not None
        and every_steps is not None
        and int(every_steps) != int(val_every)
    ):
        print(
            f"[checkpoint] monitor='{monitor}' uses validation metrics. "
            f"Keeping configured every_n_train_steps={every_steps} "
            f"(eval_every_n_steps={val_every})."
        )

    return CheckpointPolicy(
        monitor=monitor,
        mode=mode,
        filename=str(ckpt_cfg.get("filename") or _default_filename_for_monitor(monitor)),
        save_top_k=int(ckpt_cfg.get("save_top_k", 3)),
        every_n_train_steps=effective_every_steps,
    )


def build_callbacks(cfg: Dict[str, Any], checkpoint_policy: CheckpointPolicy) -> list:
    """Create Lightning callbacks, including robust fallback checkpointing."""
    ckpt_cfg = cfg.get("checkpoint", {})
    dirpath = Path(ckpt_cfg.get("dirpath", "checkpoints"))

    weighted_weights = _parse_weighted_val_accuracy_weights(ckpt_cfg.get("weights"))
    weighted_callback = _WeightedValAccuracyCallback(weights=weighted_weights)

    callbacks = [
        weighted_callback,
        ModelCheckpoint(
            dirpath=str(dirpath),
            filename=checkpoint_policy.filename,
            monitor=checkpoint_policy.monitor,
            save_top_k=checkpoint_policy.save_top_k,
            save_last=bool(ckpt_cfg.get("save_last", True)),
            every_n_train_steps=checkpoint_policy.every_n_train_steps,
            every_n_epochs=ckpt_cfg.get("every_n_epochs", 1),
            mode=checkpoint_policy.mode,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    callbacks.append(
        _EpochCheckpointCallback(
            dirpath=dirpath,
            save_top_k=checkpoint_policy.save_top_k,
            monitor=checkpoint_policy.monitor,
            mode=checkpoint_policy.mode,
        )
    )
    callbacks.append(_StartupCheckpointCallback(dirpath / "startup.ckpt"))
    callbacks.append(_InterruptCheckpointCallback(dirpath / "interrupt.ckpt"))
    return callbacks


class _WeightedValAccuracyCallback(pl.Callback):
    """Compute and expose ``weighted_val_accuracy`` from ``val_top{k}_acc``.

    On every validation epoch end, reads ``val_top{k}_acc`` entries from
    ``trainer.callback_metrics`` for the configured ``k`` values and writes
    a weighted sum back as ``weighted_val_accuracy`` so that
    ``ModelCheckpoint(monitor="weighted_val_accuracy")`` can rank checkpoints
    by it. The weights need not sum to 1.0; the metric is reported as the raw
    weighted sum, ignoring any missing top-k components for that step.
    """

    def __init__(self, weights: Mapping[int, float]) -> None:
        if not weights:
            weights = DEFAULT_WEIGHTED_VAL_ACCURACY_WEIGHTS
        self._weights: Dict[int, float] = {int(k): float(v) for k, v in weights.items()}

    @property
    def weights(self) -> Dict[int, float]:
        return dict(self._weights)

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.sanity_checking:
            return

        callback_metrics = trainer.callback_metrics
        weighted_value: float = 0.0
        contributing: int = 0

        for k, w in self._weights.items():
            metric = callback_metrics.get(f"val_top{k}_acc")
            if metric is None:
                continue
            try:
                value = float(metric)
            except (TypeError, ValueError):
                continue
            weighted_value += w * value
            contributing += 1

        if contributing == 0:
            return

        pl_module.log(
            WEIGHTED_VAL_ACCURACY_KEY,
            weighted_value,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=False,
        )
        # ``self.log`` updates callback_metrics on the next aggregation pass;
        # write directly so ModelCheckpoint, which reads callback_metrics on
        # the same hook, sees the value immediately.
        callback_metrics[WEIGHTED_VAL_ACCURACY_KEY] = torch.tensor(
            weighted_value, dtype=torch.float32
        )


class _EpochCheckpointCallback(pl.Callback):
    """Save top-k checkpoints on validation end, with a deterministic last.ckpt."""

    def __init__(
        self,
        dirpath: Path,
        save_top_k: int = 3,
        monitor: Optional[str] = None,
        mode: str = "min",
    ) -> None:
        self._dirpath = dirpath
        self._save_top_k = int(save_top_k)
        self._monitor = str(monitor or "val_loss")
        self._mode = str(mode)
        self._best = []

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking:
            return

        self._dirpath.mkdir(parents=True, exist_ok=True)
        epoch = trainer.current_epoch
        step = trainer.global_step
        metric = trainer.callback_metrics.get(self._monitor)
        metric_val: Optional[float] = None
        if metric is not None:
            try:
                metric_val = float(metric)
            except Exception:
                metric_val = None

        if metric_val is not None:
            safe_metric = self._monitor.replace("/", "_")
            filename = f"epoch{epoch:03d}-step{step}-{safe_metric}{metric_val:.4f}.ckpt"
        else:
            filename = f"epoch{epoch:03d}-step{step}.ckpt"

        ckpt_path = self._dirpath / filename
        try:
            trainer.save_checkpoint(str(ckpt_path))
            trainer.save_checkpoint(str(self._dirpath / "last.ckpt"))
        except Exception as exc:
            print(f"[epoch_ckpt] save failed: {exc}")
            return

        if metric_val is None or self._save_top_k == 0:
            return
        if self._save_top_k < 0:
            return

        self._best.append((metric_val, str(ckpt_path)))
        reverse = self._mode == "max"
        self._best.sort(key=lambda item: item[0], reverse=reverse)
        while len(self._best) > self._save_top_k:
            _, worst_path = self._best.pop()
            try:
                Path(worst_path).unlink(missing_ok=True)
            except Exception:
                pass


class _InterruptCheckpointCallback(pl.Callback):
    """Save a checkpoint on crash/interrupt via on_exception."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def on_exception(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        exception: BaseException,
    ) -> None:
        print(f"\n[interrupt] saving checkpoint -> {self._path}")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(self._path))
            print("[interrupt] checkpoint saved.")
        except Exception as exc:
            print(f"[interrupt] failed to save checkpoint: {exc}")


class _StartupCheckpointCallback(pl.Callback):
    """Save one checkpoint at training start to verify save path/write works."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        print(f"[startup] saving checkpoint -> {self._path}")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(self._path))
            print("[startup] checkpoint saved.")
        except Exception as exc:
            print(f"[startup] checkpoint save failed: {exc}")
