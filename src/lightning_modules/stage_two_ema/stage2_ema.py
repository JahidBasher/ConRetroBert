"""EMA template-encoder stabilization service for Stage-2 EMA training.

Implements the design described in EMA_DRIVEN_TEMPLATE_ENCODER_UPDATE.md:
- A deep-copied EMA shadow of the live template encoder is maintained.
- After each template optimizer step the shadow is momentum-updated.
- At epoch start the full template library is re-encoded using the EMA shadow
  so the retrieval index is built from the smoother, more stable weights.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ..stage_two.datatypes import FeatureCache, TemplateEmbeddingConfig


class Stage2EMAMixin:
    """Maintain an EMA shadow of the template encoder for stable hard-negative mining.

    This class is a standalone service injected into
    :class:`Stage2EmaLightningModule`; the name is retained for backwards
    compatibility.

    Args:
        model: Dual-encoder model with ``template_encoder`` and optional
            ``template_projector`` attributes, plus a ``_run_tower_encoder``
            helper method.
        template_input_builder: Callable that tokenizes a single template string.
        template_input_collator: Callable that batches feature dicts into
            ``Dict[str, torch.Tensor]``.
        emb_cfg: Parsed embedding cache configuration (used for encode batch size).
        decay: EMA momentum coefficient; ``ema = decay·ema + (1−decay)·live``.
    """

    def __init__(
        self,
        model: Any,
        template_input_builder: Any,
        template_input_collator: Any,
        emb_cfg: TemplateEmbeddingConfig,
        decay: float,
    ) -> None:
        self._model = model
        self._template_input_builder = template_input_builder
        self._template_input_collator = template_input_collator
        self._emb_cfg = emb_cfg
        self._decay = decay

        self._ema_encoder: Optional[torch.nn.Module] = None
        self._ema_projector: Optional[torch.nn.Module] = None

    def initialize(self, device: torch.device) -> None:
        """Deep-copy the live template encoder into EMA shadow modules.

        Must be called once from ``on_fit_start`` after the model has been
        moved to *device*.  Both shadow modules are placed on *device* and set
        to ``requires_grad=False`` / ``eval()`` mode permanently.

        Args:
            device: Device the live model currently resides on.
        """
        self._ema_encoder = copy.deepcopy(self._model.template_encoder)
        self._ema_encoder.to(device)
        self._ema_encoder.eval()
        for p in self._ema_encoder.parameters():
            p.requires_grad = False

        proj = getattr(self._model, "template_projector", None)
        if proj is not None and list(proj.parameters()):
            self._ema_projector = copy.deepcopy(proj)
            self._ema_projector.to(device)
            self._ema_projector.eval()
            for p in self._ema_projector.parameters():
                p.requires_grad = False
        else:
            self._ema_projector = None

    @torch.no_grad()
    def update_from_live(self) -> None:
        """Momentum-update EMA shadow weights from the live template encoder.

        Formula: ``ema_p = decay·ema_p + (1−decay)·live_p``

        Should be called immediately after each template optimizer step so the
        EMA always tracks the most recently updated live weights.
        """
        if self._ema_encoder is None:
            return
        for ema_p, live_p in zip(
            self._ema_encoder.parameters(),
            self._model.template_encoder.parameters(),
        ):
            ema_p.data.mul_(self._decay).add_(live_p.data, alpha=1.0 - self._decay)

        if self._ema_projector is not None:
            live_proj = getattr(self._model, "template_projector", None)
            if live_proj is not None:
                for ema_p, live_p in zip(
                    self._ema_projector.parameters(),
                    live_proj.parameters(),
                ):
                    ema_p.data.mul_(self._decay).add_(live_p.data, alpha=1.0 - self._decay)

    def encode_all_templates(
        self,
        template_list: List[str],
        device: torch.device,
        feature_cache: Optional[FeatureCache] = None,
    ) -> torch.Tensor:
        """Encode the full template library using the EMA shadow encoder.

        Uses the token-feature cache when available so templates do not need
        to be re-tokenized on each epoch rebuild.

        Args:
            template_list: Ordered list of SMARTS template strings.
            device: Device used for inference (inputs moved here per batch).
            feature_cache: Optional pre-tokenized feature tensors keyed by field
                name.  When provided, cache slices are used instead of calling
                the input builder.

        Returns:
            CPU float32 tensor of shape ``(n_templates, embed_dim)``,
            L2-normalized along the embedding dimension.
        """
        if self._ema_encoder is None:
            raise RuntimeError("EMA encoder not initialized; call initialize() first.")

        encode_bs = self._emb_cfg.encode_batch_size
        outputs: List[torch.Tensor] = []
        for start in range(0, len(template_list), encode_bs):
            inputs = self._build_batch_inputs(
                start, encode_bs, feature_cache, template_list, device
            )
            with torch.no_grad():
                _, cls = self._run_ema(inputs)
                cls = F.normalize(cls.float(), dim=-1)
            outputs.append(cls.cpu())

        return torch.cat(outputs, dim=0).contiguous()

    def encode_unique_templates(
        self,
        uniq_ids: torch.Tensor,
        template_list: List[str],
        device: torch.device,
        feature_cache: Optional[FeatureCache] = None,
    ) -> torch.Tensor:
        """Encode a selected set of templates through the EMA shadow encoder."""
        if self._ema_encoder is None:
            raise RuntimeError("EMA encoder not initialized; call initialize() first.")

        inputs = self._build_selected_inputs(uniq_ids, feature_cache, template_list, device)
        total = next(iter(inputs.values())).shape[0]
        outputs: List[torch.Tensor] = []
        for start in range(0, total, self._emb_cfg.encode_batch_size):
            end = min(start + self._emb_cfg.encode_batch_size, total)
            chunk = {k: v[start:end] for k, v in inputs.items()}
            with torch.no_grad():
                _, cls = self._run_ema(chunk)
                cls = F.normalize(cls.float(), dim=-1)
            outputs.append(cls)
        return torch.cat(outputs, dim=0).contiguous()

    def score_candidates(
        self,
        norm_product_emb: torch.Tensor,
        candidate_ids: torch.Tensor,
        template_list: List[str],
        device: torch.device,
        feature_cache: Optional[FeatureCache] = None,
    ) -> torch.Tensor:
        """Score candidates using EMA template embeddings and live product queries."""
        batch_size, cand_count = candidate_ids.shape
        flat_ids = candidate_ids.reshape(-1).detach().cpu().long()
        uniq_ids, inverse = torch.unique(flat_ids, sorted=False, return_inverse=True)
        uniq_emb = self.encode_unique_templates(
            uniq_ids, template_list, device, feature_cache
        )
        inverse = inverse.to(device=device)
        cand_emb = uniq_emb[inverse].view(batch_size, cand_count, -1)
        cand_emb = cand_emb.to(dtype=norm_product_emb.dtype)
        return (norm_product_emb.unsqueeze(1) * cand_emb).sum(dim=-1)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_batch_inputs(
        self,
        start: int,
        batch_size: int,
        feature_cache: Optional[FeatureCache],
        template_list: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if feature_cache is not None:
            inputs = {
                k: v[start : start + batch_size].to(device=device)
                for k, v in feature_cache.items()
            }
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
            return inputs

        chunk = template_list[start : start + batch_size]
        features = [self._template_input_builder(t) for t in chunk]
        inputs = self._template_input_collator(features)
        if not isinstance(inputs, dict):
            raise RuntimeError("Template input collator must return a dict.")
        return {k: v.to(device=device) for k, v in inputs.items()}

    def _build_selected_inputs(
        self,
        uniq_ids: torch.Tensor,
        feature_cache: Optional[FeatureCache],
        template_list: List[str],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if feature_cache is not None:
            inputs = {
                k: v[uniq_ids].to(device=device)
                for k, v in feature_cache.items()
            }
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].to(dtype=torch.long)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.long)
            return inputs

        texts = [template_list[int(x)] for x in uniq_ids.tolist()]
        features = [self._template_input_builder(t) for t in texts]
        inputs = self._template_input_collator(features)
        if not isinstance(inputs, dict):
            raise RuntimeError("Template input collator must return a dict.")
        return {k: v.to(device=device) for k, v in inputs.items()}

    def _run_ema(
        self, inputs: Dict[str, torch.Tensor]
    ) -> tuple:
        """Run inputs through EMA encoder and optional EMA projector."""
        hidden, cls = self._model._run_tower_encoder(self._ema_encoder, inputs)
        if self._ema_projector is not None:
            cls = self._ema_projector(cls)
        return hidden, cls
