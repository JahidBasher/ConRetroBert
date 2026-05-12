"""PyTorch Dataset classes for reaction template retrosynthesis.

Three dataset types are provided:

- :class:`ReactionDataset`: paired (product, template) data for Stage 1 training.
- :class:`Stage2ProductDataset`: product-centric data for Stage 2 template ranking.
- :class:`Stage2TemplateLibrary`: indexed collection of unique reaction templates.

Collation functions :func:`reaction_collate_fn` and :func:`stage2_collate_fn`
batch individual samples into model-ready tensors.
"""

import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset

from .datatypes import (
    FeatureCollator,
    FeatureDict,
    FilterSummary,
    InputBuilder,
    JsonRow,
    RowValidator,
)
from .input_processing import collate_feature_dicts
from .tokenizer import CharTokenizer


def iter_jsonl_rows(path: str, limit: Optional[int] = None) -> Iterable[JsonRow]:
    """Lazily yield parsed JSON objects from a JSONL file.

    Skips blank lines and stops early once *limit* rows have been yielded.

    Args:
        path: Path to a JSONL file.
        limit: Maximum number of rows to yield (None = all rows).

    Yields:
        Parsed row dicts in file order.
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                return
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1


def load_jsonl(path: str, limit: Optional[int] = None) -> List[JsonRow]:
    """Load all rows from a JSONL file into a list.

    Args:
        path: Path to a JSONL file.
        limit: Maximum number of rows to load (None = all rows).

    Returns:
        List of parsed row dicts.
    """
    rows: List[JsonRow] = []
    for row in iter_jsonl_rows(path, limit=limit):
        rows.append(row)
    return rows


def collect_template_library(path: str, limit: Optional[int] = None) -> List[str]:
    """Collect a deduplicated, first-occurrence-ordered list of templates from a JSONL file.

    Rows with missing or empty "template" fields are silently skipped.

    Args:
        path: Path to a JSONL file where each row has a "template" key.
        limit: Stop after collecting this many unique templates (None = all).

    Returns:
        Deduplicated list of SMARTS template strings in first-occurrence order.
    """
    templates: List[str] = []
    seen: Set[str] = set()
    for row in iter_jsonl_rows(path):
        tmpl = row.get("template", "")
        if not tmpl or tmpl in seen:
            continue
        seen.add(tmpl)
        templates.append(tmpl)
        if limit is not None and len(templates) >= limit:
            break
    return templates


def _encode_with_builder_or_tokenizer(
    text: str,
    kind: str,
    tokenizer: Optional[CharTokenizer],
    input_builder: Optional[InputBuilder],
    max_length: int,
    add_bos_eos: bool,
) -> FeatureDict:
    """Encode a string using an injected builder or the fallback tokenizer.

    If *input_builder* is provided it takes full priority.  Otherwise
    *tokenizer* is used with the supplied length and BOS/EOS settings.

    Args:
        text: SMILES or SMARTS string to encode.
        kind: "product" or "template" — determines the prefix token when using
            the fallback tokenizer.
        tokenizer: CharTokenizer fallback used when no builder is provided.
        input_builder: Optional pre-built callable (e.g. a HuggingFace wrapper).
        max_length: Maximum sequence length (fallback tokenizer path only).
        add_bos_eos: Whether to insert BOS/EOS tokens (fallback tokenizer path only).

    Returns:
        Feature dict with at least "input_ids" and "attention_mask".

    Raises:
        RuntimeError: If neither *input_builder* nor *tokenizer* is provided.
    """
    if input_builder is not None:
        return input_builder(text)
    if tokenizer is None:
        raise RuntimeError(
            f"No tokenizer is available and {kind}_input_builder is not configured."
        )
    return tokenizer.encode(
        text,
        kind=kind,
        max_length=max_length,
        add_bos_eos=add_bos_eos,
    )


def _add_token_backcompat(
    out: Dict[str, Any],
    features: Dict[str, Any],
    prefix: str,
) -> None:
    """Copy input_ids and attention_mask into *out* under legacy flat keys.

    Writes ``{prefix}_ids`` and ``{prefix}_mask`` into *out* as long tensors,
    converting from lists or existing tensors transparently via
    ``torch.as_tensor``.

    Args:
        out: Output dict to write the backward-compatibility keys into.
        features: Feature dict containing optional "input_ids" and
            "attention_mask" values (lists or tensors).
        prefix: Key prefix, e.g. "product" or "template".
    """
    ids = features.get("input_ids")
    if ids is not None:
        out[f"{prefix}_ids"] = torch.as_tensor(ids, dtype=torch.long)
    mask = features.get("attention_mask")
    if mask is not None:
        out[f"{prefix}_mask"] = torch.as_tensor(mask, dtype=torch.long)


def _apply_row_validator(
    row: JsonRow,
    row_validator: Optional[RowValidator],
) -> Tuple[bool, str]:
    """Apply an optional validator to a data row, swallowing validator exceptions.

    Args:
        row: Parsed JSONL row to validate.
        row_validator: Validator callable, or None (None → always keep).

    Returns:
        Tuple of (keep: bool, reason: str).
    """
    if row_validator is None:
        return True, "kept"
    try:
        keep, reason = row_validator(row)
    except Exception:
        return False, "validator_error"
    if keep:
        return True, "kept"
    return False, str(reason or "filtered")


class ReactionDataset(Dataset):
    """Stage 1 dataset of paired (product SMILES, template SMARTS) rows.

    Loads rows from a JSONL file, optionally filters them with a
    :data:`RowValidator`, and encodes each pair on demand in ``__getitem__``.
    Input encoding is fully injectable via *product_input_builder* and
    *template_input_builder*; the *tokenizer* serves as a fallback.

    Args:
        path: Path to the JSONL data file.
        tokenizer: Fallback CharTokenizer; may be None if builders are provided.
        max_product_len: Maximum token length for product encoding (fallback path).
        max_template_len: Maximum token length for template encoding (fallback path).
        add_bos_eos: Whether to add BOS/EOS tokens (fallback path only).
        limit: Maximum number of rows to load from *path*.
        product_input_builder: Injected product encoder; overrides the tokenizer.
        template_input_builder: Injected template encoder; overrides the tokenizer.
        row_validator: Optional row filter applied at load time.
    """

    def __init__(
        self,
        path: str,
        tokenizer: Optional[CharTokenizer],
        max_product_len: int,
        max_template_len: int,
        add_bos_eos: bool = False,
        limit: Optional[int] = None,
        product_input_builder: Optional[InputBuilder] = None,
        template_input_builder: Optional[InputBuilder] = None,
        row_validator: Optional[RowValidator] = None,
    ) -> None:
        self.rows: List[JsonRow] = []
        dropped_reasons: Counter = Counter()
        seen_rows = 0
        for row in iter_jsonl_rows(path, limit=limit):
            seen_rows += 1
            keep, reason = _apply_row_validator(row, row_validator)
            if not keep:
                dropped_reasons[reason] += 1
                continue
            self.rows.append(row)
        self.filter_summary: FilterSummary = {
            "enabled": row_validator is not None,
            "seen_rows": seen_rows,
            "kept_rows": len(self.rows),
            "dropped_rows": int(seen_rows - len(self.rows)),
            "dropped_reason_counts": dict(sorted(dropped_reasons.items())),
        }
        self.tokenizer = tokenizer
        self.max_product_len = max_product_len
        self.max_template_len = max_template_len
        self.add_bos_eos = add_bos_eos
        self.product_input_builder = product_input_builder
        self.template_input_builder = template_input_builder

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return encoded features for the row at position *idx*.

        Returns:
            Dict with keys: product_inputs, template_inputs, product_text,
            template_text, and legacy flat token keys (product_ids,
            product_mask, template_ids, template_mask).
        """
        row = self.rows[idx]
        product = row.get("product", "")
        template = row.get("template", "")
        prod_features = _encode_with_builder_or_tokenizer(
            text=product,
            kind="product",
            tokenizer=self.tokenizer,
            input_builder=self.product_input_builder,
            max_length=self.max_product_len,
            add_bos_eos=self.add_bos_eos,
        )
        templ_features = _encode_with_builder_or_tokenizer(
            text=template,
            kind="template",
            tokenizer=self.tokenizer,
            input_builder=self.template_input_builder,
            max_length=self.max_template_len,
            add_bos_eos=self.add_bos_eos,
        )
        out: Dict[str, Any] = {
            "product_inputs": prod_features,
            "template_inputs": templ_features,
            "product_text": product,
            "template_text": template,
        }
        _add_token_backcompat(out, prod_features, "product")
        _add_token_backcompat(out, templ_features, "template")
        return out


class Stage2TemplateLibrary:
    """Indexed collection of unique reaction templates for Stage 2 ranking.

    Provides bidirectional lookup between template SMARTS strings and integer
    IDs used throughout Stage 2 training and inference.

    Args:
        templates: Ordered list of unique SMARTS template strings.
    """

    def __init__(self, templates: List[str]) -> None:
        self.templates = templates
        self.template_to_id: Dict[str, int] = {t: i for i, t in enumerate(templates)}


def _row_pos_template_ids(
    row: JsonRow,
    template_to_id: Dict[str, int],
    n_templates: int,
) -> List[int]:
    """Extract the set of positive template IDs for a data row.

    Collects IDs from the "pos_template_ids" list field and from the
    "template" string field (if the template exists in the library),
    then deduplicates and sorts the result.

    Args:
        row: Parsed JSONL row.
        template_to_id: Mapping from template SMARTS strings to integer IDs.
        n_templates: Total number of templates in the library (used for bounds checking).

    Returns:
        Sorted list of valid non-negative template IDs; empty list if none found.
    """
    pos_ids: List[int] = []
    raw_ids = row.get("pos_template_ids")
    if isinstance(raw_ids, list):
        for x in raw_ids:
            try:
                tid = int(x)
            except Exception:
                continue
            if 0 <= tid < n_templates:
                pos_ids.append(tid)

    tmpl = row.get("template", "")
    if tmpl and tmpl in template_to_id:
        pos_ids.append(template_to_id[tmpl])

    if not pos_ids:
        return []
    return sorted(set(pos_ids))


class Stage2ProductDataset(Dataset):
    """Stage 2 dataset mapping products to positive and applicable template ID sets.

    Loads (product, template) rows from a JSONL file, optionally aggregates
    multiple rows that share the same product SMILES into a single sample, and
    merges precomputed applicable-template annotations from a separate file.
    Input encoding is injectable via *product_input_builder*.

    Args:
        path: Path to the primary JSONL data file.
        tokenizer: Fallback CharTokenizer; may be None if *product_input_builder* is set.
        template_to_id: Mapping from template SMARTS to integer IDs
            (from :class:`Stage2TemplateLibrary`).
        max_product_len: Maximum token length for product encoding (fallback path).
        add_bos_eos: Whether to add BOS/EOS tokens (fallback path only).
        limit: Maximum number of rows to read from *path*.
        aggregate_by_product: If True, merge all rows with the same product SMILES
            into one sample, unioning their positive template ID sets.
        product_input_builder: Injected product encoder; overrides the tokenizer.
        row_validator: Optional row filter applied at load time.
        appl_path: Optional path to a JSONL file with precomputed applicable
            template IDs per product.
        max_appl_per_product: Cap on applicable (non-positive) templates per product.
    """

    def __init__(
        self,
        path: str,
        tokenizer: Optional[CharTokenizer],
        template_to_id: Dict[str, int],
        max_product_len: int,
        add_bos_eos: bool = False,
        limit: Optional[int] = None,
        aggregate_by_product: bool = True,
        product_input_builder: Optional[InputBuilder] = None,
        row_validator: Optional[RowValidator] = None,
        appl_path: Optional[str] = None,
        max_appl_per_product: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_product_len = max_product_len
        self.add_bos_eos = add_bos_eos
        self.product_input_builder = product_input_builder

        n_templates = len(template_to_id)
        self.samples: List[JsonRow] = []
        dropped_reasons: Counter = Counter()
        seen_rows = 0
        kept_rows = 0

        # Load precomputed applicable template IDs keyed by product SMILES.
        product_to_appl: Dict[str, Set[int]] = defaultdict(set)
        if appl_path:
            for row in iter_jsonl_rows(appl_path):
                p = row.get("product", "")
                if not p:
                    continue
                raw = row.get("applicable_template_ids", {})
                ids: List[int] = []
                if isinstance(raw, dict):
                    for v in raw.values():
                        ids.extend(v)
                elif isinstance(raw, list):
                    ids = raw
                product_to_appl[p].update(int(t) for t in ids if 0 <= int(t) < n_templates)

        if aggregate_by_product:
            product_to_pos: Dict[str, Set[int]] = defaultdict(set)
            for row in iter_jsonl_rows(path, limit=limit):
                seen_rows += 1
                keep, reason = _apply_row_validator(row, row_validator)
                if not keep:
                    dropped_reasons[reason] += 1
                    continue
                kept_rows += 1
                product = row.get("product", "")
                if not product:
                    continue
                pos_ids = _row_pos_template_ids(row, template_to_id, n_templates)
                if not pos_ids:
                    continue
                product_to_pos[product].update(pos_ids)
            for product, pos_set in product_to_pos.items():
                # Applicable templates that are not in the positive set.
                appl_excl = sorted(product_to_appl.get(product, set()) - pos_set)[:max_appl_per_product]
                self.samples.append(
                    {
                        "product": product,
                        "pos_template_ids": sorted(pos_set),
                        "appl_template_ids": appl_excl,
                    }
                )
        else:
            for row in iter_jsonl_rows(path, limit=limit):
                seen_rows += 1
                keep, reason = _apply_row_validator(row, row_validator)
                if not keep:
                    dropped_reasons[reason] += 1
                    continue
                kept_rows += 1
                product = row.get("product", "")
                if not product:
                    continue
                pos_ids = _row_pos_template_ids(row, template_to_id, n_templates)
                if not pos_ids:
                    continue
                appl_excl = sorted(product_to_appl.get(product, set()) - set(pos_ids))[:max_appl_per_product]
                self.samples.append(
                    {
                        "product": product,
                        "pos_template_ids": pos_ids,
                        "appl_template_ids": appl_excl,
                    }
                )

        self.filter_summary: FilterSummary = {
            "enabled": row_validator is not None,
            "seen_rows": seen_rows,
            "kept_rows": kept_rows,
            "dropped_rows": int(seen_rows - kept_rows),
            "dropped_reason_counts": dict(sorted(dropped_reasons.items())),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return encoded features for the sample at position *idx*.

        Returns:
            Dict with keys: product_inputs, product_text, pos_template_ids,
            appl_template_ids, and legacy flat product token keys (product_ids,
            product_mask).
        """
        row = self.samples[idx]
        product = row["product"]
        prod_features = _encode_with_builder_or_tokenizer(
            text=product,
            kind="product",
            tokenizer=self.tokenizer,
            input_builder=self.product_input_builder,
            max_length=self.max_product_len,
            add_bos_eos=self.add_bos_eos,
        )
        out: Dict[str, Any] = {
            "product_inputs": prod_features,
            "product_text": product,
            "pos_template_ids": row["pos_template_ids"],
            "appl_template_ids": row.get("appl_template_ids", []),
        }
        _add_token_backcompat(out, prod_features, "product")
        return out


def reaction_collate_fn(
    batch: List[Dict[str, Any]],
    product_collator: FeatureCollator = collate_feature_dicts,
    template_collator: FeatureCollator = collate_feature_dicts,
) -> Dict[str, Any]:
    """Collate a batch of Stage 1 samples into model-ready tensors.

    Stacks product and template feature dicts independently, then adds
    legacy flat token keys for backward compatibility.

    Args:
        batch: List of sample dicts as returned by
            :meth:`ReactionDataset.__getitem__`.
        product_collator: Callable to batch product feature dicts.
        template_collator: Callable to batch template feature dicts.

    Returns:
        Batched dict with keys: product_inputs, template_inputs, product_text,
        template_text, and optional legacy keys (product_ids, product_mask,
        template_ids, template_mask).
    """
    product_inputs = product_collator([x["product_inputs"] for x in batch])
    template_inputs = template_collator([x["template_inputs"] for x in batch])
    out: Dict[str, Any] = {
        "product_inputs": product_inputs,
        "template_inputs": template_inputs,
        "product_text": [x["product_text"] for x in batch],
        "template_text": [x["template_text"] for x in batch],
    }
    if isinstance(product_inputs, dict):
        _add_token_backcompat(out, product_inputs, "product")
    if isinstance(template_inputs, dict):
        _add_token_backcompat(out, template_inputs, "template")
    return out


def stage2_collate_fn(
    batch: List[Dict[str, Any]],
    product_collator: FeatureCollator = collate_feature_dicts,
) -> Dict[str, Any]:
    """Collate a batch of Stage 2 samples into model-ready tensors.

    Stacks product features and aggregates template ID lists into nested lists.

    Args:
        batch: List of sample dicts as returned by
            :meth:`Stage2ProductDataset.__getitem__`.
        product_collator: Callable to batch product feature dicts.

    Returns:
        Batched dict with keys: product_inputs, product_text,
        pos_template_ids, appl_template_ids, and optional legacy product
        token keys (product_ids, product_mask).
    """
    product_inputs = product_collator([x["product_inputs"] for x in batch])
    out: Dict[str, Any] = {
        "product_inputs": product_inputs,
        "product_text": [x["product_text"] for x in batch],
        "pos_template_ids": [x["pos_template_ids"] for x in batch],
        "appl_template_ids": [x.get("appl_template_ids", []) for x in batch],
    }
    if isinstance(product_inputs, dict):
        _add_token_backcompat(out, product_inputs, "product")
    return out
