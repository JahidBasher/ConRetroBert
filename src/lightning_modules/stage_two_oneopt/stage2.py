"""One-optimizer Stage-2 module with live TE scoring and periodic bank refresh."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from ..stage_two.datatypes import CandidateSamplingConfig, FeatureCache
from ..stage_two.stage2 import Stage2LightningModule
from ..stage_two.stage2_candidate_sampler import Stage2CandidateSampler
from ..stage_two_ema.stage2 import Stage2TrainableTemplateMixin
from .datatypes import Stage2OneOptConfig


class Stage2OneOptimizerLightningModule(Stage2LightningModule):
    """Stage-2 variant that trains PE + TE with one optimizer.

    Design goals:
    - keep Lightning automatic optimization and the base single-optimizer setup
    - reuse the frozen-bank retrieval pipeline for candidate selection
    - score selected candidates with live TE so gradients flow through both towers
    - periodically refresh the template bank from the live TE at epoch boundaries
    """

    def __init__(self, cfg: Dict[str, Any], tokenizer: Any) -> None:
        super().__init__(cfg, tokenizer)

        self._oneopt_cfg = Stage2OneOptConfig.from_cfg(cfg)
        if self._cfg.freeze.alternate or self._cfg.freeze.template_frozen or self._cfg.freeze.product_frozen:
            raise RuntimeError(
                "Stage2OneOptimizerLightningModule requires "
                "freeze_template_encoder=false and freeze_product_encoder=false."
            )
        if self._oneopt_cfg.trainable_template.uses_multi_positive_contrastive():
            raise RuntimeError(
                "Stage2OneOptimizerLightningModule currently supports listwise "
                "trainable-template scoring only."
            )

        self.feature_cache: Optional[FeatureCache] = None
        self.trainable_scorer: Optional[Stage2TrainableTemplateMixin] = None
        self._configure_template_gradient_checkpointing()

    def on_fit_start(self) -> None:
        """Load template-bank assets and initialize the live-template scorer."""
        super().on_fit_start()
        self.feature_cache = self.asset_manager._load_feature_cache(self.template_list)
        self.trainable_scorer = Stage2TrainableTemplateMixin(
            model=self.model,
            template_input_builder=self.template_input_builder,
            template_input_collator=self.template_input_collator,
            trainable_cfg=self._oneopt_cfg.trainable_template,
        )
        self._configure_template_gradient_checkpointing()

    def on_train_epoch_start(self) -> None:
        """Optionally rebuild the template bank from the live TE each epoch."""
        template_frozen, product_frozen, _ = self.freeze_scheduler.apply_for_epoch(
            self.current_epoch
        )
        if template_frozen or product_frozen:
            raise RuntimeError(
                "Stage2OneOptimizerLightningModule expects PE and TE to remain trainable."
            )
        if self._should_refresh_assets(int(self.current_epoch)):
            self._refresh_assets(
                force_rebuild=self._oneopt_cfg.periodic_refresh.force_rebuild
            )
            if getattr(self.trainer, "is_global_zero", True):
                print(
                    f"[stage2_oneopt] refreshed template bank at epoch={int(self.current_epoch)}"
                )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one listwise train step with live TE scoring and one optimizer."""
        product_inputs = self._extract_product_inputs(batch)
        pos_template_ids: List[List[int]] = batch["pos_template_ids"]
        appl_template_ids: List[List[int]] = batch.get(
            "appl_template_ids", [[] for _ in pos_template_ids]
        )
        batch_has_applicable = any(len(a) > 0 for a in appl_template_ids)

        loss_cfg = self._cfg.loss

        _, product_emb = self.model.encode_product(product_inputs)
        norm_product_emb = torch.nn.functional.normalize(product_emb, dim=-1)

        cand_ids, pos_mask, scores_full_bank = self._build_training_candidates(
            norm_product_emb, pos_template_ids
        )
        total_loss, listwise_loss, entropy, _ = self._compute_train_batch_loss(
            norm_product_emb, cand_ids, pos_mask
        )

        applicable_loss = total_loss.new_tensor(0.0)
        penalty_loss = total_loss.new_tensor(0.0)
        if batch_has_applicable and self.template_emb_gpu is not None:
            allowed_negative_mask = (
                self._get_train_negative_mask(self.device)
                if self._cfg.candidate_sampling.restrict_negatives_to_train_templates
                else None
            )
            if scores_full_bank is None:
                template_bank = self.template_emb_gpu.to(dtype=norm_product_emb.dtype)
                scores_full_bank = norm_product_emb @ template_bank.t()

            if loss_cfg.applicable_loss_weight > 0:
                appl_cand_ids, appl_pos_mask = self.candidate_sampler.build_applicable_candidates_gpu(
                    scores_full=scores_full_bank,
                    appl_template_ids=appl_template_ids,
                    pos_template_ids=pos_template_ids,
                    sampling_cfg=self._cfg.candidate_sampling,
                    device=self.device,
                    allowed_negative_mask_full=allowed_negative_mask,
                )
                scores_appl = scores_full_bank.gather(1, appl_cand_ids)
                if loss_cfg.temperature > 0:
                    scores_appl = scores_appl / loss_cfg.temperature
                applicable_loss, _, _ = self._compute_listwise_loss(
                    scores_appl, appl_pos_mask
                )

            if loss_cfg.penalty_loss_weight > 0:
                penalty_loss = self._compute_penalty_loss(
                    scores_full_bank,
                    appl_template_ids,
                    pos_template_ids,
                    k=loss_cfg.penalty_top_k,
                    allowed_negative_mask_full=allowed_negative_mask,
                )

            total_loss = (
                total_loss
                + loss_cfg.applicable_loss_weight * applicable_loss
                + loss_cfg.penalty_loss_weight * penalty_loss
            )

        self.log(
            "train_listwise_loss",
            listwise_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train_entropy", entropy, on_step=True, on_epoch=True, prog_bar=False)
        self.log(
            "train_l_applicable",
            applicable_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train_l_penalty",
            penalty_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def _configure_template_gradient_checkpointing(self) -> None:
        """Enable template-encoder gradient checkpointing when requested."""
        encoder = getattr(self.model, "template_encoder", None)
        if encoder is None or not hasattr(encoder, "set_gradient_checkpointing"):
            return
        encoder.set_gradient_checkpointing(
            bool(self._oneopt_cfg.trainable_template.gradient_checkpointing)
        )

    def _should_refresh_assets(self, epoch: int) -> bool:
        cfg = self._oneopt_cfg.periodic_refresh
        if not cfg.enabled:
            return False
        if epoch < cfg.start_epoch:
            return False
        return (epoch - cfg.start_epoch) % cfg.every_n_epochs == 0

    def _refresh_assets(self, force_rebuild: bool) -> None:
        """Rebuild the template embedding cache and retrieval index from live TE."""
        self.template_emb_cpu = self.asset_manager.load_or_build_embeddings(
            self.template_list,
            self.device,
            force_rebuild=force_rebuild,
        )
        self.template_emb_gpu = (
            self.template_emb_cpu.to(self.device)
            if self._cfg.keep_embeddings_on_gpu
            else None
        )
        self.retrieval.initialize(self.template_emb_cpu)

    def _resolve_active_train_sampling_cfg(self) -> CandidateSamplingConfig:
        """Merge base candidate sampling with trainable-template overrides."""
        base = self._cfg.candidate_sampling
        override = self._oneopt_cfg.trainable_template.candidate_sampling

        raw: Dict[str, Any] = {
            "candidate_size": base.candidate_size,
            "hard_negatives": base.hard_negatives,
            "inbatch_negatives": base.inbatch_negatives,
            "random_negatives": base.random_negatives,
            "restrict_negatives_to_train_templates": bool(
                base.restrict_negatives_to_train_templates
            ),
            "retrieval": dict(base.retrieval) if base.retrieval else {},
        }

        for key, value in override.items():
            if key == "retrieval":
                continue
            raw[key] = value

        retrieval_overrides = override.get("retrieval", {})
        if isinstance(retrieval_overrides, dict):
            raw["retrieval"].update(retrieval_overrides)

        limit = self._oneopt_cfg.trainable_template.max_candidates_per_product
        if limit is not None and int(raw.get("candidate_size", base.candidate_size)) > limit:
            raw["candidate_size"] = limit

        cand_size = max(1, int(raw["candidate_size"]))
        raw["candidate_size"] = cand_size
        hard, inbatch, rand = Stage2CandidateSampler.fit_negative_budgets_to_candidate_size(
            cand_size,
            int(raw["hard_negatives"]),
            int(raw["inbatch_negatives"]),
            int(raw["random_negatives"]),
        )
        raw["hard_negatives"] = hard
        raw["inbatch_negatives"] = inbatch
        raw["random_negatives"] = rand
        return CandidateSamplingConfig.from_dict(raw)

    def _build_retrieval_rows(
        self,
        norm_product_emb: torch.Tensor,
        sampling_cfg: CandidateSamplingConfig,
        batch_size: int,
    ) -> List[List[int]]:
        """Build retrieval rows used for hard-negative candidate construction."""
        retrieval_cfg = sampling_cfg.retrieval
        rows: List[List[int]] = [[] for _ in range(batch_size)]
        if not retrieval_cfg.get("enabled", True) or sampling_cfg.hard_negatives <= 0:
            return rows
        top_k = max(
            int(retrieval_cfg.get("top_k", sampling_cfg.hard_negatives * 4)),
            sampling_cfg.hard_negatives,
        )
        allow_matrix = bool(retrieval_cfg.get("allow_matrix_in_training", False))
        return self.retrieval.retrieve(norm_product_emb, top_k, allow_matrix)

    def _should_use_gpu_candidate_builder(self) -> bool:
        if self.template_emb_gpu is None:
            return False
        override = self._oneopt_cfg.trainable_template.candidate_sampling
        if isinstance(override, dict) and "use_gpu_candidate_builder" in override:
            return bool(override["use_gpu_candidate_builder"])
        return False

    def _build_training_candidates(
        self,
        norm_product_emb: torch.Tensor,
        pos_template_ids: List[List[int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Build train candidates from the refreshed retrieval bank."""
        sampling_cfg = self._resolve_active_train_sampling_cfg()

        if self._should_use_gpu_candidate_builder():
            template_bank = self.template_emb_gpu.to(dtype=norm_product_emb.dtype)
            scores_full = norm_product_emb @ template_bank.t()
            cand_ids, pos_mask = self.candidate_sampler.build_train_candidates_gpu(
                norm_product_emb=norm_product_emb,
                pos_template_ids=pos_template_ids,
                template_emb_gpu=self.template_emb_gpu,
                sampling_cfg=sampling_cfg,
                device=self.device,
                scores_full=scores_full,
                allowed_negative_mask_full=(
                    self._get_train_negative_mask(self.device)
                    if sampling_cfg.restrict_negatives_to_train_templates
                    else None
                ),
            )
            return cand_ids, pos_mask, scores_full

        n_templates = self.template_emb_cpu.size(0)
        retrieval_rows = self._build_retrieval_rows(
            norm_product_emb, sampling_cfg, len(pos_template_ids)
        )
        cand_ids, pos_mask = self.candidate_sampler.build_train_candidates_cpu(
            pos_template_ids=pos_template_ids,
            n_templates=n_templates,
            device=self.device,
            retrieval_rows=retrieval_rows,
            candidate_size=sampling_cfg.candidate_size,
            hard_negatives=sampling_cfg.hard_negatives,
            inbatch_negatives=sampling_cfg.inbatch_negatives,
            random_negatives=sampling_cfg.random_negatives,
            allowed_negative_template_ids=(
                self._train_negative_template_ids
                if sampling_cfg.restrict_negatives_to_train_templates
                else None
            ),
        )
        return cand_ids, pos_mask, None

    def _score_candidate_templates(
        self,
        norm_product_emb: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates with live TE so gradients reach both towers."""
        raw_scores = self.trainable_scorer.score_candidates(
            norm_product_emb,
            candidate_ids,
            self.feature_cache,
            self.template_list,
            self.device,
        )
        temp = self._cfg.loss.temperature
        return raw_scores / temp if temp > 0 else raw_scores

    def _compute_train_batch_loss(
        self,
        norm_product_emb: torch.Tensor,
        cand_ids: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute listwise train loss with optional live-TE micro-batching."""
        micro_bs = self._oneopt_cfg.trainable_template.micro_batch_size
        batch_size = norm_product_emb.size(0)
        use_chunking = micro_bs is not None and micro_bs < batch_size

        if not use_chunking:
            scores = self._score_candidate_templates(norm_product_emb, cand_ids)
            total, listwise, entropy = self._compute_listwise_loss(scores, pos_mask)
            return total, listwise, entropy, scores

        cand_count = cand_ids.size(1)
        all_flat = cand_ids.reshape(-1).detach().cpu()
        global_uniq, global_inv = torch.unique(
            all_flat, sorted=False, return_inverse=True
        )
        global_uniq_emb = self.trainable_scorer.encode_unique_templates(
            global_uniq, self.feature_cache, self.template_list, self.device
        )
        global_inv = global_inv.to(device=self.device)
        global_cand_emb = global_uniq_emb[global_inv].view(batch_size, cand_count, -1)

        total = norm_product_emb.new_tensor(0.0)
        listwise = norm_product_emb.new_tensor(0.0)
        entropy_acc = norm_product_emb.new_tensor(0.0)
        score_chunks: List[torch.Tensor] = []
        temp = self._cfg.loss.temperature

        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            chunk_scores = Stage2TrainableTemplateMixin.score_from_embeddings(
                norm_product_emb[start:end], global_cand_emb[start:end]
            )
            if temp > 0:
                chunk_scores = chunk_scores / temp
            ct, cl, ce = self._compute_listwise_loss(
                chunk_scores, pos_mask[start:end]
            )
            weight = float(end - start) / float(batch_size)
            total = total + ct * weight
            listwise = listwise + cl * weight
            entropy_acc = entropy_acc + ce * weight
            score_chunks.append(chunk_scores)

        return total, listwise, entropy_acc, torch.cat(score_chunks, dim=0)
