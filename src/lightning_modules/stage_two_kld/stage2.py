"""Stage-2 KLD Lightning module.

Extends the Stage-2 EMA training path with an epoch-snapshot teacher:
- No EMA momentum updates during optimizer steps.
- At epoch start, snapshot live TE into a frozen teacher.
- Rebuild template embeddings / retrieval index from that epoch teacher.
- Apply the existing KL distillation terms against the frozen epoch teacher.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..stage_two_ema.stage2 import Stage2EmaLightningModule
from ..stage_two_ema.stage2_ema import Stage2EMAMixin
from .datatypes import Stage2KLDConfig


class Stage2KLDLightningModule(Stage2EmaLightningModule):
    """Stage-2 with epoch-snapshot teacher KLD and no EMA step-wise updates."""

    def __init__(self, cfg: Dict[str, Any], tokenizer: Any) -> None:
        super().__init__(cfg, tokenizer)
        self._kld_cfg = Stage2KLDConfig.from_cfg(cfg)
        if not self._kld_cfg.enabled:
            raise RuntimeError(
                "Stage2KLDLightningModule requires training.stage2.kld.enabled=true."
            )
        if bool(self._ema_cfg.ema.enabled):
            raise RuntimeError(
                "Stage2KLDLightningModule requires training.stage2.ema.enabled=false."
            )

    def on_fit_start(self) -> None:
        """Initialize base Stage-2 assets and build the epoch teacher service."""
        super().on_fit_start()
        if self.ema_encoder is None:
            self.ema_encoder = Stage2EMAMixin(
                model=self.model,
                template_input_builder=self.template_input_builder,
                template_input_collator=self.template_input_collator,
                emb_cfg=self._cfg.template_embeddings,
                decay=0.0,  # hard sync when update_from_live() is called
            )
            self.ema_encoder.initialize(self.device)
            if getattr(self.trainer, "is_global_zero", True):
                print(
                    f"[stage2_kld] initialized epoch teacher snapshot (device={self.device})."
                )

    def on_train_epoch_start(self) -> None:
        """Sync epoch teacher from live TE, then refresh retrieval assets."""
        for opt in (self._opt_product, self._opt_template):
            if opt is not None:
                opt.zero_grad()

        prev_template_frozen = self.freeze_scheduler._template_frozen
        template_frozen, product_frozen, changed = self.freeze_scheduler.apply_for_epoch(
            self.current_epoch
        )

        if changed and not template_frozen and prev_template_frozen is True:
            self._reset_template_optimizer()

        epoch = int(self.current_epoch)
        if (
            self._kld_cfg.sync_on_epoch_start
            and epoch >= self._kld_cfg.start_epoch
            and self.ema_encoder is not None
        ):
            self.ema_encoder.update_from_live()

        self._log_epoch_freeze_state(epoch, template_frozen, product_frozen)

        if self._should_refresh_assets(template_frozen, changed):
            # Always rebuild from the epoch teacher snapshot when refresh is active.
            self._refresh_assets(force_rebuild=True)

    def _should_refresh_assets(
        self, template_frozen: bool, freeze_state_changed: bool
    ) -> bool:
        """Refresh active epochs while skipping duplicate epoch-0 rebuild.

        ``on_fit_start`` already builds the initial embedding bank/index from the
        live model.  When ``kld.start_epoch == 0``, the epoch-teacher snapshot is
        identical to live weights at epoch start, so rebuilding again at epoch 0
        is redundant.
        """
        del template_frozen, freeze_state_changed
        epoch = int(self.current_epoch)
        if epoch < self._kld_cfg.start_epoch:
            return False
        if epoch == 0 and self._kld_cfg.start_epoch == 0:
            return False
        return True

    def _should_apply_ema_kl(self) -> bool:
        """Gate KL by epoch so KLD starts at ``kld.start_epoch``."""
        if int(self.current_epoch) < self._kld_cfg.start_epoch:
            return False
        return super()._should_apply_ema_kl()

    def _manual_backward_and_step(
        self, loss: torch.Tensor, batch_idx: int
    ) -> None:
        """Manual optimization without per-step teacher updates (no EMA behavior)."""
        accum = self._resolve_accumulation_depth()
        clip_val = float(self.cfg["training"].get("max_grad_norm", 1.0))

        self.manual_backward(loss / accum)
        if self._should_skip_for_non_finite_gradients(batch_idx):
            return

        if (batch_idx + 1) % accum != 0:
            return

        if self._opt_product is not None:
            if not self.freeze_scheduler._product_frozen:
                if clip_val > 0:
                    self.clip_gradients(self._opt_product, gradient_clip_val=clip_val)
                self._opt_product.step()
                if (
                    self._sched_product is not None
                    and self._sched_product_interval == "step"
                ):
                    self._sched_product.step()
            self._opt_product.zero_grad()

        if self._opt_template is not None:
            if self._is_template_trainable():
                if clip_val > 0:
                    self.clip_gradients(self._opt_template, gradient_clip_val=clip_val)
                self._opt_template.step()
            self._opt_template.zero_grad()

    def _log_epoch_freeze_state(
        self, epoch: int, template_frozen: bool, product_frozen: bool
    ) -> None:
        """Log freeze state with a KLD-specific prefix."""
        scoring_mode = (
            "live_template_encoding"
            if self._is_template_trainable()
            else "cached_template_embeddings"
        )
        if getattr(self.trainer, "is_global_zero", True):
            print(
                f"[stage2_kld.freeze] epoch={epoch} "
                f"template_frozen={template_frozen} product_frozen={product_frozen} "
                f"scoring_mode={scoring_mode}"
            )
        self.log("stage2_template_frozen", float(template_frozen), on_step=False, on_epoch=True)
        self.log("stage2_product_frozen", float(product_frozen), on_step=False, on_epoch=True)


# Backward-compatible alias (the top-level __init__ imports this name).
Stage2LightningModule = Stage2KLDLightningModule
