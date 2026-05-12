"""Config dataclasses for the one-optimizer Stage-2 trainable-template path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..stage_two_ema.datatypes import TrainableTemplateConfig


@dataclass
class PeriodicRefreshConfig:
    """Epoch-wise template-bank refresh settings for one-optimizer Stage-2.

    Attributes:
        enabled: Whether periodic template-bank refresh is active.
        every_n_epochs: Refresh cadence in epochs.
        start_epoch: First epoch index at which refresh is allowed.
        force_rebuild: Whether refresh bypasses any embedding cache load path and
            always re-encodes the template library from the live template encoder.
    """

    enabled: bool
    every_n_epochs: int
    start_epoch: int
    force_rebuild: bool

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PeriodicRefreshConfig":
        return cls(
            enabled=bool(raw.get("enabled", False)),
            every_n_epochs=max(1, int(raw.get("every_n_epochs", 1))),
            start_epoch=max(0, int(raw.get("start_epoch", 1))),
            force_rebuild=bool(raw.get("force_rebuild", True)),
        )


@dataclass
class Stage2OneOptConfig:
    """Config additions for the single-optimizer trainable Stage-2 path."""

    periodic_refresh: PeriodicRefreshConfig
    trainable_template: TrainableTemplateConfig

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "Stage2OneOptConfig":
        raw = cfg.get("training", {}).get("stage2", {})
        return cls(
            periodic_refresh=PeriodicRefreshConfig.from_dict(
                raw.get("one_optimizer_refresh", {})
            ),
            trainable_template=TrainableTemplateConfig.from_dict(
                raw.get("trainable_template", {})
            ),
        )
