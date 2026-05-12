"""Config dataclasses for the epoch-snapshot Stage-2 KLD path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Stage2KLDConfig:
    """Config additions for Stage-2 KLD without EMA momentum updates.

    Attributes:
        enabled: Whether Stage-2 KLD routing is active.
        sync_on_epoch_start: When True, snapshot the live template encoder into
            the teacher at each epoch start.
        start_epoch: First epoch index at which teacher-sync and refresh are
            active. Epochs before this value run without KLD teacher refresh.
    """

    enabled: bool
    sync_on_epoch_start: bool
    start_epoch: int

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "Stage2KLDConfig":
        raw = cfg.get("training", {}).get("stage2", {}).get("kld", {})
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            sync_on_epoch_start=bool(raw.get("sync_on_epoch_start", True)),
            start_epoch=max(0, int(raw.get("start_epoch", 0))),
        )
