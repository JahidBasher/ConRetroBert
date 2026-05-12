"""Stage 1 Lightning module for contrastive reaction template learning.

Trains a dual-encoder model to align product and template embeddings via
contrastive loss, with optional masked language modelling auxiliary objectives
on both towers.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ...losses import mlm_loss
from ..base import BaseConRetroLightningModule


class Stage1LightningModule(BaseConRetroLightningModule):
    """PyTorch Lightning module for Stage 1 contrastive training.

    Extends :class:`~lightning_modules.base.BaseConRetroLightningModule` with
    Stage 1-specific training and validation steps.  During validation,
    product and template embeddings are accumulated across all batches and
    epoch-level top-k retrieval accuracy is computed over the full set.

    Args:
        cfg: Full experiment config dict.
        tokenizer: CharTokenizer or compatible duck-typed encoder.
    """

    def __init__(self, cfg: Dict[str, Any], tokenizer: Any) -> None:
        super().__init__(cfg, tokenizer)
        self._val_z_p: List[torch.Tensor] = []
        self._val_z_t: List[torch.Tensor] = []

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run a single training step and return the total loss.

        Computes contrastive loss between product and template embeddings.
        When MLM is enabled in config, product and template MLM losses are
        added as auxiliary objectives weighted by ``cfg["model"]["mlm"]["weight"]``.

        Args:
            batch: Collated batch dict from :class:`~data.Stage1ReactionDataModule`.
            batch_idx: Index of the current batch within the epoch.

        Returns:
            Scalar loss tensor used for backpropagation.
        """
        use_mlm = self.cfg["model"]["mlm"]["enabled"]
        product_inputs, template_inputs, prod_labels, templ_labels = self._prepare_pair_inputs(batch, use_mlm=use_mlm)

        z_p, z_t, prod_mlm_logits, templ_mlm_logits = self.model(
            product_inputs=product_inputs, template_inputs=template_inputs
        )

        loss = self.compute_contrastive_loss(z_p, z_t)
        self.log("train_contrastive_loss", loss, on_step=True, on_epoch=True, prog_bar=False)

        if use_mlm:
            if prod_mlm_logits is not None and prod_labels is not None:
                pmlm = mlm_loss(prod_mlm_logits, prod_labels)
                loss = loss + self.mlm_weight * pmlm
                self.log("train_prod_mlm_loss", pmlm, on_step=True, on_epoch=True, prog_bar=False)
            if templ_mlm_logits is not None and templ_labels is not None:
                tmlm = mlm_loss(templ_mlm_logits, templ_labels)
                loss = loss + self.mlm_weight * tmlm
                self.log("train_templ_mlm_loss", tmlm, on_step=True, on_epoch=True, prog_bar=False)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset embedding accumulators at the start of each validation epoch."""
        self._val_z_p = []
        self._val_z_t = []

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run a single validation step and optionally accumulate embeddings.

        Computes the same loss as :meth:`training_step` (with MLM only if
        ``cfg["validation"]["mlm_eval"]`` is True).  When
        ``cfg["validation"]["compute_topk_metrics"]`` is True, raw
        (un-normalised) embeddings are accumulated for epoch-level top-k
        computation in :meth:`on_validation_epoch_end`.

        Args:
            batch: Collated batch dict from :class:`~data.Stage1ReactionDataModule`.
            batch_idx: Index of the current batch within the validation epoch.

        Returns:
            Scalar validation loss tensor.
        """
        use_mlm = self.cfg.get("validation", {}).get("mlm_eval", False) and self.cfg["model"]["mlm"]["enabled"]
        product_inputs, template_inputs, prod_labels, templ_labels = self._prepare_pair_inputs(batch, use_mlm=use_mlm)

        z_p, z_t, prod_mlm_logits, templ_mlm_logits = self.model(
            product_inputs=product_inputs, template_inputs=template_inputs
        )
        loss = self.compute_contrastive_loss(z_p, z_t)

        if use_mlm:
            if prod_mlm_logits is not None and prod_labels is not None:
                pmlm = mlm_loss(prod_mlm_logits, prod_labels)
                loss = loss + self.mlm_weight * pmlm
                self.log("val_prod_mlm_loss", pmlm, on_step=False, on_epoch=True, prog_bar=False)
            if templ_mlm_logits is not None and templ_labels is not None:
                tmlm = mlm_loss(templ_mlm_logits, templ_labels)
                loss = loss + self.mlm_weight * tmlm
                self.log("val_templ_mlm_loss", tmlm, on_step=False, on_epoch=True, prog_bar=False)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Accumulate embeddings for epoch-level top-k metrics.
        # model() returns un-normalized embeddings; normalization is done in
        # on_validation_epoch_end over the full concatenated set so that
        # top-k is computed against all val samples rather than per-batch.
        if self.cfg.get("validation", {}).get("compute_topk_metrics", False):
            self._val_z_p.append(z_p.detach())
            self._val_z_t.append(z_t.detach())

        return loss

    def on_validation_epoch_end(self) -> None:
        """Compute and log epoch-level top-k retrieval accuracy.

        Normalises the accumulated product and template embeddings, builds the
        full N×N cosine similarity matrix, and logs ``val_top{k}_acc`` for
        each k in ``cfg["validation"]["top_k"]``.  Skips silently if no
        embeddings were accumulated (e.g. top-k metrics are disabled).
        """
        if not self._val_z_p:
            return

        val_cfg = self.cfg.get("validation", {})
        top_k_list = sorted({int(k) for k in val_cfg.get("top_k", [1, 3, 5, 10, 20, 50]) if k > 0})

        # Build the full N×N cosine similarity matrix across all val samples.
        # Each paired (product_i, template_i) is the ground-truth positive for
        # product i, so top-k accuracy = fraction of products whose paired
        # template appears in the top-k retrieved templates.
        z_p = F.normalize(torch.cat(self._val_z_p, dim=0), dim=-1)  # (N, D)
        z_t = F.normalize(torch.cat(self._val_z_t, dim=0), dim=-1)  # (N, D)
        N = z_p.size(0)
        scores = z_p @ z_t.T  # (N, N); entry [i,j] = sim(product_i, template_j)

        max_k = min(max(top_k_list), N)
        _, top_idx = torch.topk(scores, k=max_k, dim=-1)  # (N, max_k)
        gt = torch.arange(N, device=scores.device).unsqueeze(1)  # (N, 1)

        for k in top_k_list:
            k_eff = min(k, max_k)
            hits = (top_idx[:, :k_eff] == gt).any(dim=1).float().mean().item()
            self.log(f"val_top{k}_acc", hits, on_epoch=True, prog_bar=(k == 1))

        self._val_z_p = []
        self._val_z_t = []
