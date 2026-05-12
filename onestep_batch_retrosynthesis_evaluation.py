#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

DEVICE = "cuda"
BATCH_SIZE = 128
TEMPLATE_TOP_K = 50
MAX_OUTCOMES_PER_TEMPLATE = 4
MAX_REACTANT_SETS = 100
EVAL_K = "1,3,5,10,20"
BREAK_ON_N = 10000000


def _format_metric_value(value, *, percent: bool):
    if value is None or value == "":
        return ""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return value
    if percent and 0.0 <= x <= 1.0:
        x *= 100.0
    return f"{x:.2f}"


def _is_percent_column(col: str) -> bool:
    if col.startswith("top") and col[3:].isdigit():
        return True
    if col.startswith("template_top") and col[len("template_top"):].isdigit():
        return True
    return (
        col.startswith("template_applicability@")
        or col.startswith("gt_template_yield_rate@")
        or col.startswith("gt_template_yield_coverage@")
    )


def _is_count_column(col: str) -> bool:
    return (
        col in {"rows_with_valid_gt", "rows_with_predictions"}
        or col.startswith("mean_unique_sets@")
        or col.startswith("gt_template_yield_count@")
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("target_dir")
    p.add_argument(
        "--templates",
        default="data/uspto-50k/merged_dataset_full.jsonl",
        help="Template library JSONL path passed to onestep_retrosynthesis_evaluation.py",
    )
    p.add_argument(
        "--eval-jsonl",
        default="data/uspto-50k/raw_test.jsonl",
        help="Evaluation JSONL path passed to onestep_retrosynthesis_evaluation.py",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--dump-all-csv",
        action="store_true",
        help="After evaluation, merge all per-experiment CSV files into one CSV under target_dir.",
    )
    p.add_argument(
        "--all-csv-name",
        default="all_experiments_eval.csv",
        help="Filename for merged CSV when --dump-all-csv is enabled.",
    )
    return p.parse_args()


def load_cfg(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) if p.suffix in {".yaml", ".yml"} else json.load(f)


def cfg_path(d: Path):
    for name in (
        "config.yaml",
        "config.yml",
        "config.json",
        "hparams.yaml",
        "hparams.yml",
    ):
        p = d / name
        if p.exists():
            return p
    return None


def has_ckpt(d: Path) -> bool:
    return any(d.glob("*.ckpt"))


def exp_dirs(root: Path):
    if cfg_path(root) and has_ckpt(root):
        return [root]
    return sorted(
        d for d in root.rglob("*") if d.is_dir() and cfg_path(d) and has_ckpt(d)
    )


def cmd(repo: Path, cfg: Path, ckpt: Path, templates: str, eval_jsonl: str):
    return [
        sys.executable,
        str(repo / "onestep_retrosynthesis_evaluation.py"),
        "--config",
        str(cfg),
        "--checkpoint",
        str(ckpt),
        "--templates",
        templates,
        "--eval_jsonl",
        eval_jsonl,
        "--template-top-k",
        str(TEMPLATE_TOP_K),
        "--device",
        DEVICE,
        "--max-outcomes-per-template",
        str(MAX_OUTCOMES_PER_TEMPLATE),
        "--max-reactant-sets",
        str(MAX_REACTANT_SETS),
        "--eval-k",
        EVAL_K,
        "--batch-size",
        str(BATCH_SIZE),
        "--summary",
        "--summary-out",
        "summary.json",
        "--rebuild_cache",
        "--break_on_n",
        str(BREAK_ON_N),
    ]


def dump_excel(exp_dir: Path):
    out = exp_dir / f"{exp_dir.name}.csv"
    cols = [
        "experiment",
        "checkpoint",
        "generated_at",
        "eval_jsonl",
        "rows_with_valid_gt",
        "rows_with_predictions",
        "top1",
        "top3",
        "top5",
        "top10",
        "top20",
        "template_top1",
        "template_top3",
        "template_top5",
        "template_top10",
        "template_top20",
        "template_applicability@1",
        "template_applicability@3",
        "template_applicability@5",
        "template_applicability@10",
        "template_applicability@20",
        "mean_unique_sets@1",
        "mean_unique_sets@3",
        "mean_unique_sets@5",
        "mean_unique_sets@10",
        "mean_unique_sets@20",
        "gt_template_yield_count@1",
        "gt_template_yield_count@3",
        "gt_template_yield_count@5",
        "gt_template_yield_count@10",
        "gt_template_yield_count@20",
        "gt_template_yield_rate@1",
        "gt_template_yield_rate@3",
        "gt_template_yield_rate@5",
        "gt_template_yield_rate@10",
        "gt_template_yield_rate@20",
        "gt_template_yield_coverage@1",
        "gt_template_yield_coverage@3",
        "gt_template_yield_coverage@5",
        "gt_template_yield_coverage@10",
        "gt_template_yield_coverage@20",
    ]
    rows = [cols]
    for s in sorted(exp_dir.glob("*/summary.json")):
        d = json.load(s.open("r", encoding="utf-8"))
        r = d.get("run_info", {})
        pct = lambda key: _format_metric_value(d.get(key), percent=True)
        cnt = lambda key: _format_metric_value(d.get(key), percent=False)
        rows.append(
            [
                s.parent.name,
                Path(r.get("checkpoint", "")).name,
                r.get("generated_at"),
                r.get("eval_jsonl"),
                cnt("rows_with_valid_gt"),
                cnt("rows_with_predictions"),
                pct("top1_accuracy"),
                pct("top3_accuracy"),
                pct("top5_accuracy"),
                pct("top10_accuracy"),
                pct("top20_accuracy"),
                pct("template_top1_accuracy"),
                pct("template_top3_accuracy"),
                pct("template_top5_accuracy"),
                pct("template_top10_accuracy"),
                pct("template_top20_accuracy"),
                pct("template_applicability_rate@1"),
                pct("template_applicability_rate@3"),
                pct("template_applicability_rate@5"),
                pct("template_applicability_rate@10"),
                pct("template_applicability_rate@20"),
                cnt("mean_unique_reactant_sets_per_product@1"),
                cnt("mean_unique_reactant_sets_per_product@3"),
                cnt("mean_unique_reactant_sets_per_product@5"),
                cnt("mean_unique_reactant_sets_per_product@10"),
                cnt("mean_unique_reactant_sets_per_product@20"),
                cnt("gt_template_yield_count@1"),
                cnt("gt_template_yield_count@3"),
                cnt("gt_template_yield_count@5"),
                cnt("gt_template_yield_count@10"),
                cnt("gt_template_yield_count@20"),
                pct("gt_template_yield_rate@1"),
                pct("gt_template_yield_rate@3"),
                pct("gt_template_yield_rate@5"),
                pct("gt_template_yield_rate@10"),
                pct("gt_template_yield_rate@20"),
                pct("gt_template_yield_coverage@1"),
                pct("gt_template_yield_coverage@3"),
                pct("gt_template_yield_coverage@5"),
                pct("gt_template_yield_coverage@10"),
                pct("gt_template_yield_coverage@20"),
            ]
        )
    with out.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[run_eval] wrote {out}")


def dump_all_csv(root: Path, dirs, out_name: str):
    csv_paths = [d / f"{d.name}.csv" for d in dirs if (d / f"{d.name}.csv").exists()]
    if not csv_paths:
        print("[run_eval] no per-experiment CSV files found to merge")
        return None

    all_cols = []
    rows = []
    for p in csv_paths:
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not all_cols:
                all_cols = list(reader.fieldnames or [])
                if "experiment_dir" not in all_cols:
                    all_cols = ["experiment_dir"] + all_cols
            for row in reader:
                out_row = {k: row.get(k) for k in all_cols if k != "experiment_dir"}
                out_row["experiment_dir"] = str(p.parent)
                for col in all_cols:
                    if col in {"experiment_dir", "experiment", "checkpoint", "generated_at", "eval_jsonl"}:
                        continue
                    if col not in out_row:
                        continue
                    if _is_percent_column(col):
                        out_row[col] = _format_metric_value(out_row[col], percent=True)
                    elif _is_count_column(col):
                        out_row[col] = _format_metric_value(out_row[col], percent=False)
                rows.append(out_row)

    if not rows:
        print("[run_eval] no rows found while merging per-experiment CSV files")
        return None

    out = root / out_name
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[run_eval] wrote merged CSV {out}")
    return out


def main():
    args = parse_args()
    repo = Path(__file__).resolve().parent
    root = Path(args.target_dir).resolve()
    dirs = exp_dirs(root)
    if not dirs:
        raise RuntimeError(f"No experiment folders found under: {root}")
    failures = 0
    for d in dirs:
        out = d / f"{d.name}.csv"
        if out.exists():
            print(f"[run_eval] skipping {d} since {out} already exists\n\n\n")
            continue

        print(f"[run_eval] processing experiment directory: {d}")

        cfg = cfg_path(d)
        templates = args.templates
        eval_jsonl = args.eval_jsonl
        ckpts = sorted(p for p in d.glob("*.ckpt") if p.is_file())
        if not (templates and eval_jsonl and ckpts):
            print(
                f"[run_eval] skipping {d} templates={templates} eval={eval_jsonl} ckpts={len(ckpts)}"
            )
            failures += 1
            continue
        print(f"[run_eval] experiment={d} config={cfg} ckpts={len(ckpts)}")
        for ckpt in ckpts:
            c = cmd(repo, cfg, ckpt, str(templates), str(eval_jsonl))
            print(" ".join(c))
            if not args.dry_run:
                try:
                    subprocess.run(c, cwd=repo, check=True)
                except subprocess.CalledProcessError:
                    failures += 1
                    print(f"[run_eval] failed {ckpt}")
        if not args.dry_run:
            dump_excel(d)

    if args.dump_all_csv:
        dump_all_csv(root, dirs, args.all_csv_name)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
