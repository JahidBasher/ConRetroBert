import argparse
from typing import Any, Dict, Optional

import pytorch_lightning as pl

from src.utils import (
    load_config,
    set_seed,
    load_or_build_tokenizer,
    load_weights_into_lightning_module,
)
from src.lightning_modules import create_lightning_module
from src.data import create_data_module
from src.train_utils import (
    build_callbacks,
    build_logger,
    configure_runtime,
    resolve_artifact_dir,
    resolve_checkpoint_policy,
    save_config_snapshot,
    save_tracked_code_snapshot,
    warn_if_stage_and_config_mismatch,
)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for config selection and checkpoint initialization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2])
    parser.add_argument("--resume", default=None, help="Path to Lightning checkpoint for resume")
    parser.add_argument("--pretrained", default=None, help="Path to model-only or Lightning checkpoint")
    return parser.parse_args()


def _resolve_pretrained_path(
    cfg: Dict[str, Any],
    cli_pretrained: Optional[str],
    cli_resume: Optional[str],
) -> Optional[str]:
    """Return the weight initialization path, applying Stage-2 fallback rules.

    When ``--pretrained`` is not given on the CLI, falls back to
    ``training.stage2.init_checkpoint`` from config.  Raises if Stage-2
    requires a checkpoint but neither source provides one.

    Args:
        cfg: Full experiment config dict.
        cli_pretrained: Value of ``--pretrained`` from the CLI (may be None).
        cli_resume: Value of ``--resume`` from the CLI (may be None).

    Returns:
        Resolved path to use for weight initialization, or None.
    """
    stage = int(cfg["training"]["stage"])
    stage2_cfg = cfg["training"].get("stage2", {})

    pretrained = cli_pretrained or stage2_cfg.get("init_checkpoint") or None

    if stage == 2 and bool(stage2_cfg.get("require_stage1_checkpoint", True)):
        if cli_resume is None and pretrained is None:
            raise RuntimeError(
                "Stage 2 requires initialization from a Stage 1 checkpoint. "
                "Provide --pretrained or training.stage2.init_checkpoint."
            )

    return pretrained


def _build_trainer(
    cfg: Dict[str, Any],
    logger: Any,
    callbacks: list,
    use_auto_opt: bool,
) -> pl.Trainer:
    """Construct the PyTorch Lightning Trainer from experiment config.

    Manual-optimization modules (e.g., Stage-2 EMA) handle gradient clipping
    and accumulation internally, so those Trainer arguments are suppressed when
    ``automatic_optimization`` is False.

    Args:
        cfg: Full experiment config dict.
        logger: Configured Lightning logger instance.
        callbacks: List of Lightning callbacks.
        use_auto_opt: Whether the module uses automatic optimization.

    Returns:
        Configured :class:`pl.Trainer` instance.
    """
    tcfg = cfg["training"]
    log_cfg = cfg.get("logging", {})
    val_every = cfg.get("validation", {}).get("eval_every_n_steps")

    return pl.Trainer(
        max_epochs=tcfg["epochs"],
        accelerator=log_cfg.get("accelerator", "auto"),
        devices=log_cfg.get("devices", "auto"),
        precision=log_cfg.get("precision", "32-true"),
        log_every_n_steps=log_cfg.get("log_every_n_steps", 10),
        enable_progress_bar=log_cfg.get("progress_bar", True),
        gradient_clip_val=tcfg.get("max_grad_norm", 1.0) if use_auto_opt else None,
        accumulate_grad_batches=tcfg.get("accumulate_grad_batches", 1) if use_auto_opt else 1,
        val_check_interval=val_every if val_every is not None else 1.0,
        logger=logger,
        callbacks=callbacks,
    )


def main() -> None:
    """Run end-to-end training setup and launch PyTorch Lightning fit."""
    args = _parse_args()

    cfg = load_config(args.config)
    if args.stage is not None:
        cfg["training"]["stage"] = args.stage

    pretrained_path = _resolve_pretrained_path(cfg, args.pretrained, args.resume)

    stage = int(cfg["training"]["stage"])
    set_seed(cfg.get("seed", 42))
    configure_runtime(cfg.get("logging", {}))

    # --- Build core objects ---
    tokenizer = load_or_build_tokenizer(cfg, allow_build=True)
    datamodule = create_data_module(cfg, tokenizer)
    module = create_lightning_module(cfg, tokenizer)

    if pretrained_path:
        load_weights_into_lightning_module(module, pretrained_path)

    # --- Artifact setup ---
    warn_if_stage_and_config_mismatch(args.config, stage)
    artifact_dir = resolve_artifact_dir(cfg=cfg, stage=stage)
    print(f"[artifact] writing run outputs to {artifact_dir}")

    log_cfg = cfg.get("logging", {})
    logger = build_logger(log_cfg=log_cfg, stage=stage, artifact_dir=artifact_dir)
    save_config_snapshot(cfg=cfg, artifact_dir=artifact_dir)

    snapshot_path = save_tracked_code_snapshot(artifact_dir=artifact_dir)
    if snapshot_path:
        print(f"[artifact] saved project snapshot to {snapshot_path}")
    else:
        print("[artifact] warning: could not save project snapshot")

    # --- Callbacks + trainer ---
    ckpt_cfg = cfg.get("checkpoint", {})
    checkpoint_policy = resolve_checkpoint_policy(
        cfg=cfg,
        has_val=bool(cfg["data"].get("val_path")),
        val_every=cfg.get("validation", {}).get("eval_every_n_steps"),
        every_steps=ckpt_cfg.get("every_n_train_steps"),
    )
    callbacks = build_callbacks(cfg=cfg, checkpoint_policy=checkpoint_policy)

    use_auto_opt = bool(getattr(module, "automatic_optimization", True))
    trainer = _build_trainer(cfg, logger, callbacks, use_auto_opt)

    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
