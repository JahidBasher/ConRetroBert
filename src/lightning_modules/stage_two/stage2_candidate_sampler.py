"""Stage-2 candidate set construction for training and validation.

All GPU and CPU paths for building positive/negative candidate sets live here.
The sampler is stateless except for a seeded RNG used in random negative
sampling; it can be safely shared across training and validation steps.
"""

import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

from .datatypes import CandidateRow, CandidateSamplingConfig, PositiveMaskRow


class Stage2CandidateSampler:
    """Candidate-set construction helpers for Stage-2 training and validation.

    Provides three concrete candidate builders:

    - :meth:`build_train_candidates_gpu`: fully GPU-resident construction using
      the precomputed full score matrix.
    - :meth:`build_train_candidates_cpu`: CPU construction from retrieval rows,
      in-batch negatives, and random fill.
    - :meth:`build_eval_candidates_gpu` / :meth:`build_eval_candidates_cpu`:
      evaluation variants that preserve all positives in the candidate set.

    Args:
        seed: Integer seed for the internal Python RNG used in random negative
            sampling.  Reproducible across runs when set deterministically.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._trimmed_positive_warning_emitted = False
        self._gpu_train_padding_warning_emitted = False
        self._gpu_applicable_padding_warning_emitted = False

    # ------------------------------------------------------------------
    # Positive normalisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_positive_template_ids(
        raw_pos_ids: List[int],
        n_templates: int,
        candidate_size: Optional[int],
        context: str,
    ) -> List[int]:
        """Filter positives to valid IDs and enforce non-empty, feasible sets.

        Args:
            raw_pos_ids: Raw (possibly out-of-range) positive template IDs.
            n_templates: Total number of templates in the library.
            candidate_size: Maximum allowed candidate set size; positives must
                fit within it (None = no check).
            context: Human-readable label for error messages (e.g. "Stage 2 train").

        Returns:
            Sorted, deduplicated list of valid positive template IDs.

        Raises:
            RuntimeError: If no valid positives remain after filtering, or if
                the positives exceed *candidate_size*.
        """
        positives = sorted({int(x) for x in raw_pos_ids if 0 <= int(x) < n_templates})
        if not positives:
            raise RuntimeError(
                f"{context} sample without positive template ids after filtering."
            )
        if candidate_size is not None and len(positives) > candidate_size:
            raise RuntimeError(
                f"{context} positives exceed candidate_size; increase candidate_size."
            )
        return positives

    # ------------------------------------------------------------------
    # CPU random sampling helpers
    # ------------------------------------------------------------------

    def sample_from_candidate_pool(self, pool: List[int], k: int) -> List[int]:
        """Sample up to *k* IDs from *pool* without replacement.

        Args:
            pool: Pool of integer candidate IDs to sample from.
            k: Maximum number of IDs to return.

        Returns:
            List of sampled IDs; may be shorter than *k* if the pool is small.
        """
        if k <= 0 or not pool:
            return []
        if len(pool) <= k:
            return list(pool)
        return self._rng.sample(pool, k)

    def normalize_train_positive_template_ids(
        self,
        raw_pos_ids: List[int],
        n_templates: int,
        candidate_size: int,
        context: str,
        max_positive_fraction: float = 0.33,
    ) -> List[int]:
        """Normalize training positives and trim oversized sets to preserve negatives.

        Training candidate sets need some room for negatives. When a sample has
        more positives than the configured budget allows, keep a random subset of
        positives capped at ``floor(candidate_size * max_positive_fraction)`` and
        leave the remaining slots for hard/in-batch/random negatives.

        Args:
            raw_pos_ids: Raw positive template IDs for one sample.
            n_templates: Total number of templates in the library.
            candidate_size: Maximum candidate set width.
            context: Human-readable label for warning messages.
            max_positive_fraction: Maximum fraction of the candidate budget that
                can be occupied by positives.

        Returns:
            Sorted, deduplicated list of valid positive template IDs, trimmed when
            necessary to preserve some negative capacity.
        """
        positives = self.normalize_positive_template_ids(
            raw_pos_ids=raw_pos_ids,
            n_templates=n_templates,
            candidate_size=None,
            context=context,
        )
        candidate_size = max(1, int(candidate_size))
        max_positive_count = max(1, int(candidate_size * float(max_positive_fraction)))
        if candidate_size > 1:
            max_positive_count = min(max_positive_count, candidate_size - 1)
        else:
            max_positive_count = 1

        if len(positives) <= max_positive_count:
            return positives

        trimmed = sorted(self._rng.sample(positives, max_positive_count))
        if not self._trimmed_positive_warning_emitted:
            print(
                f"{context} positives exceeded the training positive budget; "
                f"trimming to {max_positive_count} positives per sample "
                f"({max_positive_fraction:.0%} of candidate_size) to preserve negatives."
            )
            self._trimmed_positive_warning_emitted = True
        return trimmed

    def sample_random_template_ids(
        self,
        n_templates: int,
        k: int,
        blocked: Set[int],
        allowed_template_ids: Optional[Sequence[int]] = None,
        allow_repeats_when_exhausted: bool = False,
    ) -> List[int]:
        """Sample up to *k* random template IDs while avoiding IDs in *blocked*.

        Modifies *blocked* in-place by adding each sampled ID.

        Args:
            n_templates: Total number of templates (upper bound for sampling).
            k: Maximum number of IDs to return.
            blocked: Set of IDs to exclude; updated in-place.

        Returns:
            List of newly sampled template IDs (may be shorter than *k* if
            the available population is small).
        """
        if k <= 0:
            return []

        if allowed_template_ids is not None:
            allowed_unique = sorted(
                {int(t) for t in allowed_template_ids if 0 <= int(t) < n_templates}
            )
            if not allowed_unique:
                return []
            available = [tid for tid in allowed_unique if tid not in blocked]
            if len(available) >= k:
                sampled = self._rng.sample(available, k)
                blocked.update(sampled)
                return sampled
            sampled = list(available)
            blocked.update(sampled)
            if not allow_repeats_when_exhausted:
                return sampled
            seed_pool = sampled if sampled else allowed_unique
            while len(sampled) < k:
                sampled.append(self._rng.choice(seed_pool))
            return sampled

        sampled: List[int] = []
        tries = 0
        max_tries = max(10 * k, 100)
        while len(sampled) < k and tries < max_tries:
            template_id = self._rng.randrange(0, n_templates)
            if template_id not in blocked:
                blocked.add(template_id)
                sampled.append(template_id)
            tries += 1
        return sampled

    def sample_with_replacement(self, pool: List[int], k: int) -> List[int]:
        """Sample *k* IDs from *pool* with replacement."""
        if k <= 0 or not pool:
            return []
        return [self._rng.choice(pool) for _ in range(k)]

    @staticmethod
    def _pad_candidate_ids_with_row_repeats(
        cand_ids: torch.Tensor,
        target_size: int,
    ) -> torch.Tensor:
        """Pad candidate IDs to *target_size* by repeating existing row entries."""
        current = cand_ids.size(1)
        if current >= target_size:
            return cand_ids[:, :target_size]
        batch = cand_ids.size(0)
        if current <= 0:
            return torch.zeros(batch, target_size, dtype=torch.long, device=cand_ids.device)
        deficit = target_size - current
        repeat_index = torch.randint(
            low=0,
            high=current,
            size=(batch, deficit),
            device=cand_ids.device,
        )
        repeats = cand_ids.gather(1, repeat_index)
        return torch.cat([cand_ids, repeats], dim=1)

    @staticmethod
    def fit_negative_budgets_to_candidate_size(
        candidate_size: int,
        hard_negatives: int,
        inbatch_negatives: int,
        random_negatives: int,
    ) -> Tuple[int, int, int]:
        """Clamp negative budgets so the total fits within *candidate_size*.

        Candidate rows must keep at least one slot for positives, so the total
        negative budget is capped at ``candidate_size - 1``.  Allocation priority
        is: hard negatives > in-batch negatives > random negatives.

        Args:
            candidate_size: Maximum total candidates per sample.
            hard_negatives: Requested hard negative count.
            inbatch_negatives: Requested in-batch negative count.
            random_negatives: Requested random negative count.

        Returns:
            ``(hard, inbatch, random)`` clamped to fit within the budget.
        """
        cap = max(0, int(candidate_size) - 1)
        hard = max(0, int(hard_negatives))
        inbatch = max(0, int(inbatch_negatives))
        rand = max(0, int(random_negatives))

        if hard + inbatch + rand <= cap:
            return hard, inbatch, rand

        hard_keep = min(hard, cap)
        remaining = cap - hard_keep
        inbatch_keep = min(inbatch, remaining)
        random_keep = min(rand, remaining - inbatch_keep)
        return hard_keep, inbatch_keep, random_keep

    @staticmethod
    def collect_in_batch_negative_candidates(
        row_idx: int,
        all_pos: List[List[int]],
        blocked: Set[int],
    ) -> List[int]:
        """Collect other samples' positives as an in-batch negative pool.

        Args:
            row_idx: Index of the current sample (excluded from collection).
            all_pos: Positive ID lists for all samples in the batch.
            blocked: IDs already assigned to the current sample's candidate set.

        Returns:
            Sorted, deduplicated list of in-batch negative candidate IDs.
        """
        pool: List[int] = []
        for other_idx, other_pos in enumerate(all_pos):
            if other_idx == row_idx:
                continue
            pool.extend(tid for tid in other_pos if tid not in blocked)
        return sorted(set(pool))

    @staticmethod
    def finalize_candidate_row(
        cand: CandidateRow,
        pos: List[int],
        candidate_size: int,
    ) -> Tuple[CandidateRow, PositiveMaskRow]:
        """Trim the candidate row to *candidate_size* and produce the positive mask.

        Positives are always retained; excess candidates are removed from the
        non-positive tail first.

        Args:
            cand: Full candidate list (positives + negatives).
            pos: Positive template IDs for this sample.
            candidate_size: Target width of the candidate set.

        Returns:
            Tuple of (trimmed_candidates, positive_mask).

        Raises:
            RuntimeError: If all positives were excluded from the candidate set.
        """
        pos_set = set(pos)
        if len(cand) > candidate_size:
            non_pos = [tid for tid in cand if tid not in pos_set]
            cand = pos + non_pos[: candidate_size - len(pos)]

        pos_mask = [tid in pos_set for tid in cand]
        if not any(pos_mask):
            raise RuntimeError("Positive templates were excluded from candidate set.")
        return cand, pos_mask

    # ------------------------------------------------------------------
    # Shared tensor utilities
    # ------------------------------------------------------------------

    @staticmethod
    def build_positive_template_mask(
        batch_pos_sets: List[List[int]],
        n_templates: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a dense boolean positive mask of shape ``[B, N]`` via scatter.

        Args:
            batch_pos_sets: Positive template ID lists, one per sample.
            n_templates: Total number of templates ``N``.
            device: Target device for the output tensor.

        Returns:
            Boolean tensor of shape ``[B, N]`` where ``mask[i, j]`` is True
            iff template *j* is a positive for sample *i*.
        """
        batch_size = len(batch_pos_sets)
        mask = torch.zeros(batch_size, n_templates, dtype=torch.bool, device=device)

        batch_idx: List[int] = []
        template_idx: List[int] = []
        for i, positives in enumerate(batch_pos_sets):
            batch_idx.extend([i] * len(positives))
            template_idx.extend(positives)

        if batch_idx:
            bi = torch.tensor(batch_idx, dtype=torch.long, device=device)
            ti = torch.tensor(template_idx, dtype=torch.long, device=device)
            mask[bi, ti] = True
        return mask

    @staticmethod
    def to_candidate_tensors(
        candidates: List[CandidateRow],
        pos_masks: List[PositiveMaskRow],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert candidate rows and masks from Python lists to tensors.

        Args:
            candidates: List of candidate ID rows (must all have the same length).
            pos_masks: List of boolean mask rows aligned with *candidates*.
            device: Target device for the output tensors.

        Returns:
            Tuple of ``(candidate_ids [B, C], positive_mask [B, C])``.
        """
        return (
            torch.tensor(candidates, dtype=torch.long, device=device),
            torch.tensor(pos_masks, dtype=torch.bool, device=device),
        )

    @staticmethod
    def _pad_or_trim_candidate_ids(
        cand_ids: torch.Tensor, candidate_size: int
    ) -> torch.Tensor:
        """Return candidate IDs with exact width *candidate_size*.

        Trims from the right if wider; zero-pads on the right if narrower.

        Args:
            cand_ids: Candidate ID tensor of shape ``[B, W]``.
            candidate_size: Target width ``C``.

        Returns:
            Tensor of shape ``[B, candidate_size]``.
        """
        total = cand_ids.size(1)
        if total > candidate_size:
            return cand_ids[:, :candidate_size]
        if total < candidate_size:
            pad = cand_ids.new_zeros(cand_ids.size(0), candidate_size - total)
            return torch.cat([cand_ids, pad], dim=1)
        return cand_ids

    @staticmethod
    def _sample_random_ids_from_allowed_mask(
        allowed_mask: torch.Tensor,
        k: int,
        *,
        device: torch.device,
        dtype_source: torch.Tensor,
    ) -> torch.Tensor:
        """Uniformly sample *k* IDs per row from allowed positions via top-k noise.

        Assigns a uniform random score to each allowed position and -1.0 to
        blocked positions, then selects the top-*k* by score.

        Args:
            allowed_mask: Boolean mask ``[B, N]`` where True = position is eligible.
            k: Number of IDs to sample per row.
            device: Device for temporary tensors.
            dtype_source: Tensor used to determine fill dtype for blocked positions.

        Returns:
            Sampled ID tensor of shape ``[B, k]``.
        """
        batch_size, template_count = allowed_mask.shape
        noise = torch.where(
            allowed_mask,
            torch.rand(batch_size, template_count, device=device),
            dtype_source.new_full((batch_size, template_count), -1.0),
        )
        _, sampled_ids = torch.topk(noise, k=k, dim=-1)
        return sampled_ids

    # ------------------------------------------------------------------
    # GPU candidate builders
    # ------------------------------------------------------------------

    def build_train_candidates_gpu(
        self,
        norm_product_emb: torch.Tensor,
        pos_template_ids: List[List[int]],
        template_emb_gpu: torch.Tensor,
        sampling_cfg: CandidateSamplingConfig,
        device: torch.device,
        scores_full: Optional[torch.Tensor] = None,
        allowed_negative_mask_full: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct Stage-2 training candidates fully on GPU.

        Selects hard negatives (top scoring non-positives) and random negatives
        from the full score matrix, then pads/trims to *candidate_size*.

        Args:
            norm_product_emb: L2-normalised product embeddings ``[B, D]``.
            pos_template_ids: Positive template ID lists, one per sample.
            template_emb_gpu: L2-normalised template embeddings ``[N, D]`` on GPU.
            sampling_cfg: Parsed candidate sampling configuration.
            device: Target device.
            scores_full: Optional precomputed score matrix ``[B, N]``.  If
                provided, avoids an extra matmul.

        Returns:
            Tuple of ``(candidate_ids [B, C], positive_mask [B, C])``.
        """
        template_count = template_emb_gpu.size(0)
        candidate_size = max(1, int(sampling_cfg.candidate_size))
        hard_negatives, inbatch_negatives, random_negatives = (
            self.fit_negative_budgets_to_candidate_size(
                candidate_size=candidate_size,
                hard_negatives=int(sampling_cfg.hard_negatives),
                inbatch_negatives=int(sampling_cfg.inbatch_negatives),
                random_negatives=int(sampling_cfg.random_negatives),
            )
        )

        batch_pos_sets = [
            self.normalize_train_positive_template_ids(
                raw, template_count, candidate_size, "Stage 2 train"
            )
            for raw in pos_template_ids
        ]
        pos_mask_full = self.build_positive_template_mask(batch_pos_sets, template_count, device)
        scores = (
            scores_full
            if scores_full is not None
            else norm_product_emb @ template_emb_gpu.to(dtype=norm_product_emb.dtype).t()
        )

        if allowed_negative_mask_full is not None:
            allowed_negative_mask_full = allowed_negative_mask_full.to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            )
        n_structured = min(hard_negatives + inbatch_negatives, template_count)
        structured_scores = scores.masked_fill(pos_mask_full, float("inf"))
        if allowed_negative_mask_full is not None:
            disallowed_negatives = (~allowed_negative_mask_full).unsqueeze(0) & (~pos_mask_full)
            structured_scores = structured_scores.masked_fill(disallowed_negatives, float("-inf"))
            if n_structured > 0:
                per_row_available = (
                    ((~pos_mask_full) & allowed_negative_mask_full.unsqueeze(0)).sum(dim=-1)
                    + pos_mask_full.sum(dim=-1)
                )
                n_structured = min(n_structured, int(per_row_available.min().item()))

        if n_structured > 0:
            _, structured_ids = torch.topk(structured_scores, k=n_structured, dim=-1)
        else:
            structured_ids = torch.empty(scores.size(0), 0, dtype=torch.long, device=device)

        n_random = min(random_negatives, template_count - n_structured)
        random_excluded = pos_mask_full.clone()
        if structured_ids.numel() > 0:
            random_excluded.scatter_(1, structured_ids, True)
        if allowed_negative_mask_full is not None:
            random_excluded |= (~allowed_negative_mask_full).unsqueeze(0)
        if n_random > 0:
            max_common_rand = int((~random_excluded).sum(dim=-1).min().item())
            n_random = min(n_random, max_common_rand)
        if n_random > 0:
            random_ids = self._sample_random_ids_from_allowed_mask(
                ~random_excluded,
                n_random,
                device=device,
                dtype_source=scores,
            )
            cand_ids = torch.cat([structured_ids, random_ids], dim=1)
        else:
            cand_ids = structured_ids

        pre_pad_width = cand_ids.size(1)
        if pre_pad_width < candidate_size and not self._gpu_train_padding_warning_emitted:
            print(
                "Stage 2 train candidates were shorter than candidate_size; "
                f"repeat-padding from width {pre_pad_width} to {candidate_size}."
            )
            self._gpu_train_padding_warning_emitted = True
        cand_ids = self._pad_candidate_ids_with_row_repeats(cand_ids, candidate_size)
        positive_mask = pos_mask_full.gather(1, cand_ids)
        return cand_ids, positive_mask

    def build_applicable_candidates_gpu(
        self,
        scores_full: torch.Tensor,
        appl_template_ids: List[List[int]],
        pos_template_ids: List[List[int]],
        sampling_cfg: CandidateSamplingConfig,
        device: torch.device,
        allowed_negative_mask_full: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build candidates for the L_applicable term (applicable positives + non-applicable negatives).

        Positives are the applicable templates (appl ∪ pos); negatives are the
        highest-scoring non-applicable templates plus random fills.

        Args:
            scores_full: Precomputed score matrix ``[B, N]``.
            appl_template_ids: Precomputed applicable template IDs per sample.
            pos_template_ids: Observed positive template IDs per sample.
            sampling_cfg: Parsed candidate sampling configuration.
            device: Target device.

        Returns:
            Tuple of ``(candidate_ids [B, C], positive_mask [B, C])``.
        """
        batch_size, template_count = scores_full.shape
        candidate_size = sampling_cfg.candidate_size
        hard_negatives = sampling_cfg.hard_negatives
        random_negatives = sampling_cfg.random_negatives

        combined_applicable = [
            sorted(set(appl) | set(pos))
            for appl, pos in zip(appl_template_ids, pos_template_ids)
        ]
        applicable_mask_full = self.build_positive_template_mask(
            combined_applicable, template_count, device
        )

        if allowed_negative_mask_full is not None:
            allowed_negative_mask_full = allowed_negative_mask_full.to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            )

        negative_scores = scores_full.masked_fill(applicable_mask_full, float("-inf"))
        if allowed_negative_mask_full is not None:
            negative_scores = negative_scores.masked_fill(
                (~allowed_negative_mask_full).unsqueeze(0), float("-inf")
            )
        n_hard = min(hard_negatives, template_count)
        if allowed_negative_mask_full is not None and n_hard > 0:
            allowed_nonappl = (~applicable_mask_full) & allowed_negative_mask_full.unsqueeze(0)
            max_common = int(allowed_nonappl.sum(dim=-1).min().item())
            n_hard = min(n_hard, max_common)
        if n_hard > 0:
            _, hard_ids = torch.topk(negative_scores, k=n_hard, dim=-1)
        else:
            hard_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        random_excluded = applicable_mask_full.clone()
        random_excluded.scatter_(1, hard_ids, True)
        if allowed_negative_mask_full is not None:
            random_excluded |= (~allowed_negative_mask_full).unsqueeze(0)
        n_rand = min(random_negatives, template_count)
        if n_rand > 0:
            allowed_for_random = ~random_excluded
            max_common_rand = int(allowed_for_random.sum(dim=-1).min().item())
            n_rand = min(n_rand, max_common_rand)
        if n_rand > 0:
            rand_ids = self._sample_random_ids_from_allowed_mask(
                ~random_excluded,
                n_rand,
                device=device,
                dtype_source=scores_full,
            )
        else:
            rand_ids = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        appl_only = [list(set(appl)) for appl in appl_template_ids]
        max_appl = max((len(row) for row in appl_only), default=0)

        if max_appl > 0:
            appl_tensor = torch.zeros(
                batch_size, max_appl, dtype=torch.long, device=device
            )
            for i, ids in enumerate(appl_only):
                if ids:
                    appl_tensor[i, : len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=device
                    )

            neg_ids = torch.cat([hard_ids, rand_ids], dim=1)
            cand_ids = torch.cat([appl_tensor, neg_ids], dim=1)
        else:
            # No applicable positives in this batch — return negatives only.
            cand_ids = torch.cat([hard_ids, rand_ids], dim=1)

        pre_pad_width = cand_ids.size(1)
        if pre_pad_width < candidate_size and not self._gpu_applicable_padding_warning_emitted:
            print(
                "Stage 2 applicable candidates were shorter than candidate_size; "
                f"repeat-padding from width {pre_pad_width} to {candidate_size}."
            )
            self._gpu_applicable_padding_warning_emitted = True
        cand_ids = self._pad_candidate_ids_with_row_repeats(cand_ids, candidate_size)
        positive_mask = applicable_mask_full.gather(1, cand_ids)
        return cand_ids, positive_mask

    def build_eval_candidates_gpu(
        self,
        scores_full: torch.Tensor,
        pos_template_ids: List[List[int]],
        candidate_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Construct GPU validation candidates and masks from a full score matrix.

        Selects the top-*candidate_size* templates by score (positives are
        guaranteed to be retained via masked fill).

        Args:
            scores_full: Full score matrix ``[B, N]``.
            pos_template_ids: Positive template ID lists per sample.
            candidate_size: Number of candidates to select.

        Returns:
            Tuple of ``(candidate_ids [B, C], positive_mask [B, C],
            full_positive_mask [B, N])``.
        """
        _, template_count = scores_full.shape
        device = scores_full.device
        candidate_size = min(candidate_size, template_count)

        batch_pos_sets = [
            self.normalize_positive_template_ids(
                raw, template_count, candidate_size, "Stage 2 validation"
            )
            for raw in pos_template_ids
        ]
        full_positive_mask = self.build_positive_template_mask(
            batch_pos_sets, template_count, device
        )
        # Inflate positive scores so they are always selected first.
        scores_with_pos_boosted = scores_full.masked_fill(full_positive_mask, float("inf"))
        _, cand_ids = torch.topk(scores_with_pos_boosted, k=candidate_size, dim=-1)
        positive_mask = full_positive_mask.gather(1, cand_ids)
        return cand_ids, positive_mask, full_positive_mask

    # ------------------------------------------------------------------
    # CPU candidate builders
    # ------------------------------------------------------------------

    def build_train_candidates_cpu(
        self,
        pos_template_ids: List[List[int]],
        n_templates: int,
        device: torch.device,
        retrieval_rows: List[List[int]],
        candidate_size: int,
        hard_negatives: int,
        inbatch_negatives: int,
        random_negatives: int,
        allowed_negative_template_ids: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct CPU training candidates from retrieval + in-batch + random pools.

        For each sample:
        1. Start with positives.
        2. Add up to *hard_negatives* from the retrieval result.
        3. Fill with in-batch negatives from other samples' positives.
        4. Fill remaining capacity with random negatives.

        Args:
            pos_template_ids: Positive template ID lists per sample.
            n_templates: Total number of templates in the library.
            device: Target device for output tensors.
            retrieval_rows: Top-k retrieved template IDs per sample (from the
                retrieval index).
            candidate_size: Total candidate set width.
            hard_negatives: Maximum hard negatives from retrieval.
            inbatch_negatives: Maximum in-batch negatives.
            random_negatives: Maximum random negatives.

        Returns:
            Tuple of ``(candidate_ids [B, C], positive_mask [B, C])``.
        """
        candidate_size = max(1, int(candidate_size))
        hard_negatives, inbatch_negatives, random_negatives = (
            self.fit_negative_budgets_to_candidate_size(
                candidate_size=candidate_size,
                hard_negatives=hard_negatives,
                inbatch_negatives=inbatch_negatives,
                random_negatives=random_negatives,
            )
        )

        batch_pos_sets = [
            self.normalize_train_positive_template_ids(
                raw, n_templates, candidate_size, "Stage 2 train"
            )
            for raw in pos_template_ids
        ]
        allowed_negative_pool = (
            tuple(
                sorted(
                    {
                        int(t)
                        for t in allowed_negative_template_ids
                        if 0 <= int(t) < n_templates
                    }
                )
            )
            if allowed_negative_template_ids is not None
            else None
        )
        allowed_negative_set = (
            set(allowed_negative_pool) if allowed_negative_pool is not None else None
        )

        candidate_rows: List[CandidateRow] = []
        positive_masks: List[PositiveMaskRow] = []
        for row_idx, pos in enumerate(batch_pos_sets):
            pos_set = set(pos)
            cand = list(pos)
            assigned_ids: Set[int] = set(cand)

            hard_added = 0
            for template_id in retrieval_rows[row_idx]:
                if (
                    allowed_negative_set is not None
                    and template_id not in allowed_negative_set
                ):
                    continue
                if template_id in assigned_ids:
                    continue
                cand.append(template_id)
                assigned_ids.add(template_id)
                hard_added += 1
                if hard_added >= hard_negatives:
                    break

            in_batch_pool = self.collect_in_batch_negative_candidates(
                row_idx=row_idx, all_pos=batch_pos_sets, blocked=assigned_ids
            )
            if allowed_negative_set is not None:
                in_batch_pool = [tid for tid in in_batch_pool if tid in allowed_negative_set]
            for template_id in self.sample_from_candidate_pool(in_batch_pool, inbatch_negatives):
                if template_id not in assigned_ids:
                    cand.append(template_id)
                    assigned_ids.add(template_id)

            cand.extend(
                self.sample_random_template_ids(
                    n_templates,
                    random_negatives,
                    assigned_ids,
                    allowed_template_ids=allowed_negative_pool,
                    allow_repeats_when_exhausted=allowed_negative_pool is not None,
                )
            )
            if len(cand) < candidate_size:
                cand.extend(
                    self.sample_random_template_ids(
                        n_templates,
                        candidate_size - len(cand),
                        assigned_ids,
                        allowed_template_ids=allowed_negative_pool,
                        allow_repeats_when_exhausted=allowed_negative_pool is not None,
                    )
                )
            if len(cand) < candidate_size and allowed_negative_set is not None:
                filler_pool = [tid for tid in cand if tid not in pos_set]
                if not filler_pool:
                    filler_pool = [int(pos[0])] if pos else [0]
                cand.extend(
                    self.sample_with_replacement(
                        filler_pool, candidate_size - len(cand)
                    )
                )

            finalized_cand, positive_mask = self.finalize_candidate_row(
                cand=cand, pos=pos, candidate_size=candidate_size
            )
            candidate_rows.append(finalized_cand)
            positive_masks.append(positive_mask)

        return self.to_candidate_tensors(candidate_rows, positive_masks, device)

    def build_eval_candidates_cpu(
        self,
        pos_template_ids: List[List[int]],
        n_templates: int,
        device: torch.device,
        retrieval_rows: List[List[int]],
        candidate_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct CPU validation candidates from retrieval rows with random fill.

        Starts with positives, fills from retrieval results, then pads with
        random negatives to reach *candidate_size*.

        Args:
            pos_template_ids: Positive template ID lists per sample.
            n_templates: Total number of templates in the library.
            device: Target device for output tensors.
            retrieval_rows: Top-k retrieved template IDs per sample.
            candidate_size: Total candidate set width.

        Returns:
            Tuple of ``(candidate_ids [B, C], positive_mask [B, C])``.
        """
        candidate_rows: List[CandidateRow] = []
        positive_masks: List[PositiveMaskRow] = []
        for row_idx, raw_pos in enumerate(pos_template_ids):
            pos = self.normalize_positive_template_ids(
                raw_pos, n_templates, candidate_size, "Stage 2 validation"
            )
            pos_set = set(pos)
            cand = list(pos)

            for template_id in retrieval_rows[row_idx]:
                if template_id in pos_set:
                    continue
                cand.append(template_id)
                if len(cand) >= candidate_size:
                    break

            if len(cand) < candidate_size:
                assigned_ids = set(cand)
                cand.extend(
                    self.sample_random_template_ids(
                        n_templates, candidate_size - len(cand), assigned_ids
                    )
                )

            finalized_cand, positive_mask = self.finalize_candidate_row(
                cand=cand, pos=pos, candidate_size=candidate_size
            )
            candidate_rows.append(finalized_cand)
            positive_masks.append(positive_mask)

        return self.to_candidate_tensors(candidate_rows, positive_masks, device)
