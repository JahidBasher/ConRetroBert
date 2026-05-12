"""RDKit-based chemistry validation for (product, template) dataset rows.

Validates that a SMARTS reaction template produces at least one valid
reactant set when applied to the product molecule via RDKit.  Used to
filter low-quality training rows before model training.

The module also exposes a CLI entry point (``python -m src.data.validation``)
for running validation over a full JSONL dataset file.
"""

import argparse
import functools
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .datatypes import JsonRow, RowValidator


def _require_rdkit() -> Tuple[Any, Any]:
    """Import and return RDKit modules, raising ImportError if unavailable.

    Returns:
        Tuple of (rdkit.Chem, rdkit.Chem.rdChemReactions).

    Raises:
        ImportError: If RDKit is not installed in the current environment.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdChemReactions
    except Exception as exc:
        raise ImportError("RDKit is required for data validation.") from exc
    return Chem, rdChemReactions


def _safe_pct(n: int, d: int) -> float:
    """Compute a percentage safely, returning 0.0 when the denominator is zero.

    Args:
        n: Numerator count.
        d: Denominator count.

    Returns:
        100 * n / d, or 0.0 if d <= 0.
    """
    if d <= 0:
        return 0.0
    return 100.0 * float(n) / float(d)


def _analyze_product_template_pair(
    product_smiles: str,
    template_smarts: str,
) -> Dict[str, Any]:
    """Run RDKit template application and collect outcome statistics.

    Applies *template_smarts* to *product_smiles* as a retrosynthetic
    reaction and counts raw, valid, and unique reactant outcomes.

    Args:
        product_smiles: SMILES string of the target molecule.
        template_smarts: SMARTS/SMIRKS string of the reaction template.

    Returns:
        Dict with keys: is_valid (bool), reason (str), raw_outcome_count (int),
        valid_outcome_count (int), unique_outcome_count (int),
        min_reactant_count (int).
    """
    Chem, rdChemReactions = _require_rdkit()

    mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        return {
            "is_valid": False,
            "reason": "invalid_product_smiles",
            "raw_outcome_count": 0,
            "valid_outcome_count": 0,
            "unique_outcome_count": 0,
            "min_reactant_count": 0,
        }

    try:
        rxn = rdChemReactions.ReactionFromSmarts(template_smarts)
    except Exception:
        rxn = None
    if rxn is None:
        return {
            "is_valid": False,
            "reason": "invalid_template_smarts",
            "raw_outcome_count": 0,
            "valid_outcome_count": 0,
            "unique_outcome_count": 0,
            "min_reactant_count": 0,
        }

    try:
        reactant_sets = rxn.RunReactants((mol,))
    except Exception:
        return {
            "is_valid": False,
            "reason": "run_reactants_error",
            "raw_outcome_count": 0,
            "valid_outcome_count": 0,
            "unique_outcome_count": 0,
            "min_reactant_count": 0,
        }

    valid_outcomes: List[Tuple[str, ...]] = []
    for reactants in reactant_sets:
        smi = []
        ok = True
        for r in reactants:
            if r is None:
                ok = False
                break
            s = Chem.MolToSmiles(r, canonical=True)
            if not s:
                ok = False
                break
            smi.append(s)
        if ok:
            valid_outcomes.append(tuple(smi))

    unique_outcomes = sorted({tuple(sorted(x)) for x in valid_outcomes})
    if not unique_outcomes:
        return {
            "is_valid": False,
            "reason": "no_valid_outcomes",
            "raw_outcome_count": int(len(reactant_sets)),
            "valid_outcome_count": int(len(valid_outcomes)),
            "unique_outcome_count": 0,
            "min_reactant_count": 0,
        }

    min_reactant_count = min(len(out) for out in unique_outcomes)
    return {
        "is_valid": True,
        "reason": "valid",
        "raw_outcome_count": int(len(reactant_sets)),
        "valid_outcome_count": int(len(valid_outcomes)),
        "unique_outcome_count": int(len(unique_outcomes)),
        "min_reactant_count": int(min_reactant_count),
    }


def validate_product_template_row(
    row: JsonRow,
    product_field: str = "product",
    template_field: str = "template",
) -> Tuple[bool, str]:
    """Validate a single data row by checking template applicability via RDKit.

    Extracts the product SMILES and template SMARTS from *row*, then runs
    RDKit reaction application to confirm the template produces at least one
    valid reactant outcome.

    Args:
        row: A parsed JSONL data row.
        product_field: Key in *row* holding the product SMILES string.
        template_field: Key in *row* holding the template SMARTS string.

    Returns:
        Tuple of (keep: bool, reason: str).
    """
    product = row.get(product_field)
    template = row.get(template_field)

    if not isinstance(product, str) or not product.strip():
        return False, "missing_or_invalid_product"
    if not isinstance(template, str) or not template.strip():
        return False, "missing_or_invalid_template"

    result = _analyze_product_template_pair(
        product_smiles=product,
        template_smarts=template,
    )
    if bool(result.get("is_valid", False)):
        return True, "valid"
    return False, str(result.get("reason", "unknown_invalid"))


def build_row_validator_from_config(
    cfg: Dict[str, Any],
    split: str,
) -> Optional[RowValidator]:
    """Build a row validator from the experiment config for a given data split.

    Reads ``cfg["data"]["validation_filter"]`` to determine whether validation
    is enabled, which splits it applies to, and which row fields to use.
    The returned callable is a ``functools.partial`` over
    :func:`validate_product_template_row` bound to the configured field names.

    Args:
        cfg: Full experiment config dict.
        split: Data split name (e.g. "train", "val", "test").

    Returns:
        A :data:`RowValidator` callable if validation is enabled for this
        split, or None if all rows should be accepted without filtering.

    Raises:
        RuntimeError: If an unsupported ``validation_filter.mode`` is set.
    """
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    validation_cfg = data_cfg.get("validation_filter", {})
    if not isinstance(validation_cfg, dict):
        return None
    if not bool(validation_cfg.get("enabled", False)):
        return None

    apply_splits = validation_cfg.get("apply_splits", ["train", "val", "test"])
    if isinstance(apply_splits, str):
        apply_splits = [apply_splits]
    apply_set = {str(x).lower() for x in apply_splits}
    split_norm = str(split).lower()
    if "all" not in apply_set and split_norm not in apply_set:
        return None

    mode = str(validation_cfg.get("mode", "product_template_applicability")).lower()
    if mode != "product_template_applicability":
        raise RuntimeError(
            "Unsupported data.validation_filter.mode. Use 'product_template_applicability'."
        )

    product_field = str(validation_cfg.get("product_field", "product"))
    template_field = str(validation_cfg.get("template_field", "template"))

    return functools.partial(
        validate_product_template_row,
        product_field=product_field,
        template_field=template_field,
    )


def validate_dataset_jsonl(
    path: str,
    product_field: str = "product",
    template_field: str = "template",
    limit: Optional[int] = None,
    max_failure_examples: int = 20,
    progress_every: int = 0,
) -> Dict[str, Any]:
    """Validate every row in a JSONL file and return aggregated statistics.

    For each parsed row the (product, template) pair is checked for RDKit
    applicability.  Aggregate counts, rates, histograms, and failure examples
    are collected and returned as a summary dict.

    Args:
        path: Path to a JSONL file where each line is a JSON object.
        product_field: Key holding the product SMILES in each row.
        template_field: Key holding the template SMARTS in each row.
        limit: Stop after this many successfully parsed rows (None = no limit).
        max_failure_examples: Maximum number of failing row details to collect.
        progress_every: Print a progress line every N rows (0 = disabled).

    Returns:
        Summary dict with validation statistics, reason counts, and
        failure examples.
    """
    _require_rdkit()

    total_lines = 0
    non_empty_lines = 0
    parsed_rows = 0
    checked_rows = 0
    valid_rows = 0
    invalid_rows = 0

    reason_counts: Counter = Counter()
    min_reactant_count_hist: Counter = Counter()

    raw_outcome_sum = 0
    valid_outcome_sum = 0
    unique_outcome_sum = 0
    multi_outcome_rows = 0
    max_unique_outcomes = 0

    failure_examples: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            total_lines += 1
            if limit is not None and checked_rows >= limit:
                break

            line = line.strip()
            if not line:
                continue
            non_empty_lines += 1

            try:
                row = json.loads(line)
            except Exception:
                reason_counts["invalid_json"] += 1
                if len(failure_examples) < max_failure_examples:
                    failure_examples.append({"line": line_no, "reason": "invalid_json"})
                continue

            if not isinstance(row, dict):
                reason_counts["row_not_object"] += 1
                if len(failure_examples) < max_failure_examples:
                    failure_examples.append({"line": line_no, "reason": "row_not_object"})
                continue

            parsed_rows += 1
            checked_rows += 1

            product = row.get(product_field)
            template = row.get(template_field)

            if not isinstance(product, str) or not product.strip():
                reason = "missing_or_invalid_product"
                invalid_rows += 1
                reason_counts[reason] += 1
                if len(failure_examples) < max_failure_examples:
                    failure_examples.append({"line": line_no, "reason": reason})
                continue

            if not isinstance(template, str) or not template.strip():
                reason = "missing_or_invalid_template"
                invalid_rows += 1
                reason_counts[reason] += 1
                if len(failure_examples) < max_failure_examples:
                    failure_examples.append({"line": line_no, "reason": reason, "product": product})
                continue

            result = _analyze_product_template_pair(
                product_smiles=product,
                template_smarts=template,
            )

            if result["is_valid"]:
                valid_rows += 1
                raw_outcome_sum += int(result["raw_outcome_count"])
                valid_outcome_sum += int(result["valid_outcome_count"])
                unique_outcome_sum += int(result["unique_outcome_count"])
                if int(result["unique_outcome_count"]) > 1:
                    multi_outcome_rows += 1
                max_unique_outcomes = max(max_unique_outcomes, int(result["unique_outcome_count"]))
                min_reactant_count_hist[int(result["min_reactant_count"])] += 1
            else:
                invalid_rows += 1
                reason = str(result["reason"])
                reason_counts[reason] += 1
                if len(failure_examples) < max_failure_examples:
                    failure_examples.append(
                        {
                            "line": line_no,
                            "reason": reason,
                            "product": product,
                            "template": template,
                        }
                    )

            if progress_every > 0 and checked_rows % progress_every == 0:
                print(f"[progress] checked={checked_rows} valid={valid_rows} invalid={invalid_rows}")

    summary: Dict[str, Any] = {
        "input_path": path,
        "total_lines": total_lines,
        "non_empty_lines": non_empty_lines,
        "parsed_rows": parsed_rows,
        "checked_rows": checked_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "valid_rate_percent": _safe_pct(valid_rows, checked_rows),
        "invalid_rate_percent": _safe_pct(invalid_rows, checked_rows),
        "failure_reason_counts": dict(reason_counts),
        "valid_rows_with_multiple_unique_outcomes": multi_outcome_rows,
        "max_unique_outcomes_in_row": max_unique_outcomes,
        "avg_raw_outcomes_per_valid_row": (float(raw_outcome_sum) / float(valid_rows)) if valid_rows > 0 else 0.0,
        "avg_valid_outcomes_per_valid_row": (float(valid_outcome_sum) / float(valid_rows)) if valid_rows > 0 else 0.0,
        "avg_unique_outcomes_per_valid_row": (float(unique_outcome_sum) / float(valid_rows)) if valid_rows > 0 else 0.0,
        "min_reactant_count_distribution": dict(sorted(min_reactant_count_hist.items())),
        "failure_examples": failure_examples,
    }
    return summary


def _print_validation_summary(summary: Dict[str, Any]) -> None:
    """Print a human-readable summary of dataset validation results to stdout.

    Args:
        summary: Dict as returned by :func:`validate_dataset_jsonl`.
    """
    print("=== Dataset Template Applicability Summary ===")
    print(f"input_path: {summary['input_path']}")
    print(f"total_lines: {summary['total_lines']}")
    print(f"non_empty_lines: {summary['non_empty_lines']}")
    print(f"parsed_rows: {summary['parsed_rows']}")
    print(f"checked_rows: {summary['checked_rows']}")
    print(f"valid_rows: {summary['valid_rows']} ({summary['valid_rate_percent']:.2f}%)")
    print(f"invalid_rows: {summary['invalid_rows']} ({summary['invalid_rate_percent']:.2f}%)")
    print("")

    print("failure_reason_counts:")
    reasons = summary.get("failure_reason_counts", {})
    if reasons:
        for k in sorted(reasons.keys()):
            print(f"  {k}: {reasons[k]}")
    else:
        print("  (none)")
    print("")

    print("valid_outcome_stats:")
    print(f"  avg_raw_outcomes_per_valid_row: {summary['avg_raw_outcomes_per_valid_row']:.4f}")
    print(f"  avg_valid_outcomes_per_valid_row: {summary['avg_valid_outcomes_per_valid_row']:.4f}")
    print(f"  avg_unique_outcomes_per_valid_row: {summary['avg_unique_outcomes_per_valid_row']:.4f}")
    print(f"  valid_rows_with_multiple_unique_outcomes: {summary['valid_rows_with_multiple_unique_outcomes']}")
    print(f"  max_unique_outcomes_in_row: {summary['max_unique_outcomes_in_row']}")
    print("")

    print("min_reactant_count_distribution:")
    hist = summary.get("min_reactant_count_distribution", {})
    if hist:
        for reactant_count in sorted(hist.keys(), key=lambda x: int(x)):
            print(f"  {reactant_count}: {hist[reactant_count]}")
    else:
        print("  (none)")
    print("")

    print("failure_examples:")
    examples = summary.get("failure_examples", [])
    if not examples:
        print("  (none)")
    else:
        for ex in examples:
            line = ex.get("line")
            reason = ex.get("reason")
            product = ex.get("product")
            template = ex.get("template")
            print(f"  line={line} reason={reason}")
            if product:
                print(f"    product={product}")
            if template:
                print(f"    template={template}")


def _main() -> None:
    """CLI entry point: validate a JSONL dataset file and print a summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate JSONL dataset rows by checking whether each (product, template) "
            "pair can generate at least one reactant outcome via RDKit reaction application."
        )
    )
    parser.add_argument("--input", required=True, help="Path to JSONL dataset file.")
    parser.add_argument("--product_field", default="product", help="JSON field name for product SMILES.")
    parser.add_argument("--template_field", default="template", help="JSON field name for template SMARTS/SMIRKS.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of parsed rows to check.")
    parser.add_argument(
        "--max_failure_examples", type=int, default=20, help="Number of failing row examples to include."
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=0,
        help="Print progress every N checked rows (0 disables progress logging).",
    )
    parser.add_argument("--summary_out", default=None, help="Optional output JSON path for full summary.")
    args = parser.parse_args()

    summary = validate_dataset_jsonl(
        path=args.input,
        product_field=args.product_field,
        template_field=args.template_field,
        limit=args.limit,
        max_failure_examples=args.max_failure_examples,
        progress_every=args.progress_every,
    )
    _print_validation_summary(summary)

    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("")
        print(f"Wrote summary JSON to: {args.summary_out}")


if __name__ == "__main__":
    _main()
