"""Validation metric computation for Stage-2 top-k retrieval evaluation."""

from typing import Callable, Dict, List

import torch

from .datatypes import EvaluationConfig


class Stage2MetricsMixin:
    """Compute top-k retrieval accuracy and recall for Stage-2 validation.

    Provides two evaluation paths:

    - :meth:`compute_gpu`: Fully vectorized over a precomputed ``[B, N]`` score
      matrix; no Python per-sample loops.
    - :meth:`compute_cpu`: Python loops over per-sample retrieval results from an
      external retrieve function (used when GPU embeddings are unavailable).

    This class is a standalone service injected into :class:`Stage2LightningModule`;
    the name is retained for backwards compatibility.

    Args:
        eval_cfg: Parsed evaluation configuration controlling top-k thresholds
            and whether metrics are computed at all.
    """

    def __init__(self, eval_cfg: EvaluationConfig) -> None:
        self._eval_cfg = eval_cfg

    def compute_gpu(
        self,
        scores_full: torch.Tensor,
        pos_mask_full: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute top-k accuracy/recall from a full GPU similarity matrix.

        Uses ``topk`` + cumulative hit counting over the precomputed positive mask,
        avoiding any Python-level per-sample iteration.

        Args:
            scores_full: Raw similarity scores of shape ``(B, N)`` — one score
                per template for each sample in the batch.
            pos_mask_full: Boolean positive mask of shape ``(B, N)``; ``True``
                where the template is a ground-truth positive.

        Returns:
            Dict mapping metric names to scalar floats (e.g. ``val_top5_acc``).
            Empty dict when top-k metrics are disabled or all samples lack positives.
        """
        if not self._eval_cfg.compute_topk_metrics:
            return {}

        top_k_list = sorted({int(k) for k in self._eval_cfg.top_k if int(k) > 0})
        if not top_k_list:
            return {}

        batch_size, n_templates = scores_full.shape
        max_k = min(max(top_k_list), n_templates)

        valid_mask = pos_mask_full.any(dim=1)
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return {}

        pos_counts = pos_mask_full.sum(dim=1).clamp_min(1).float()
        _, top_idx = torch.topk(scores_full, k=max_k, dim=-1)
        top_is_pos = pos_mask_full.gather(1, top_idx).float()
        cum_hits = top_is_pos.cumsum(dim=1)

        metrics: Dict[str, float] = {
            "val_positive_coverage": float(valid_count) / float(max(1, batch_size))
        }
        for k in top_k_list:
            k_idx = min(k, max_k) - 1
            hits = cum_hits[:, k_idx]
            valid_hits = hits[valid_mask]
            valid_pos_counts = pos_counts[valid_mask]
            metrics[f"val_top{k}_acc"] = (valid_hits > 0).float().mean().item()
            metrics[f"val_top{k}_recall"] = (valid_hits / valid_pos_counts).mean().item()

        return metrics

    def compute_cpu(
        self,
        norm_product_emb: torch.Tensor,
        pos_template_ids: List[List[int]],
        n_templates: int,
        retrieve_fn: Callable[[torch.Tensor, int, bool], List[List[int]]],
    ) -> Dict[str, float]:
        """Compute top-k accuracy/recall using ranked candidates from CPU retrieval.

        Iterates over samples in Python, checking how many of the ground-truth
        positives appear in the top-k retrieved templates.

        Args:
            norm_product_emb: L2-normalized product embeddings, shape ``(B, D)``.
            pos_template_ids: Per-sample lists of positive template integer IDs.
            n_templates: Total number of templates in the library.
            retrieve_fn: Callable with signature
                ``(z_p_norm, top_k, allow_matrix) -> List[List[int]]`` that returns
                ranked template IDs for each sample (e.g. ``retrieval.retrieve``).

        Returns:
            Dict mapping metric names to scalar floats.  Empty dict when top-k
            metrics are disabled or no sample has a valid positive.
        """
        if not self._eval_cfg.compute_topk_metrics:
            return {}

        top_k_list = sorted({int(k) for k in self._eval_cfg.top_k if int(k) > 0})
        if not top_k_list:
            return {}

        max_k = min(max(top_k_list), n_templates)
        ranked_ids = retrieve_fn(norm_product_emb, max_k, True)

        valid_count = 0
        total_count = len(pos_template_ids)
        acc_sums = {k: 0.0 for k in top_k_list}
        recall_sums = {k: 0.0 for k in top_k_list}

        for i, raw_pos in enumerate(pos_template_ids):
            pos = sorted({int(x) for x in raw_pos if 0 <= int(x) < n_templates})
            if not pos:
                continue
            valid_count += 1
            pos_set = set(pos)
            ranked = ranked_ids[i]
            for k in top_k_list:
                topk = ranked[: min(k, len(ranked))]
                hits = sum(1 for tid in topk if tid in pos_set)
                acc_sums[k] += 1.0 if hits > 0 else 0.0
                recall_sums[k] += float(hits) / float(len(pos_set))

        if valid_count == 0:
            return {}

        metrics: Dict[str, float] = {
            "val_positive_coverage": float(valid_count) / float(max(1, total_count))
        }
        for k in top_k_list:
            metrics[f"val_top{k}_acc"] = acc_sums[k] / valid_count
            metrics[f"val_top{k}_recall"] = recall_sums[k] / valid_count
        return metrics
