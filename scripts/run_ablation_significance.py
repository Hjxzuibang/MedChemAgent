"""Ablation significance: full agent vs each single-component ablation.

Reads results/ablation.json (best_agg lists per target/variant) and runs a
one-sided Mann-Whitney U test per (target, ablation) with the hypothesis that
the full agent's best aggregate is higher. Each observation is one completed
repeat. Writes results/ablation_significance.txt in the same cell format as
results/benchmark_significance.txt.
"""
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]

ABLATIONS = ("no_rag", "no_tools", "no_feedback")
TARGET_ORDER = ("EGFR", "CDK2", "ABL1", "BRAF", "VEGFR2", "SRC")
DEFAULT_INPUT = "results/ablation.json"


def resolve_path(path_string):
    """Resolve relative paths from the repository root, not the caller's CWD."""
    path = Path(path_string)
    return path if path.is_absolute() else ROOT / path


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return "NA"
    sd = arr.std(ddof=1) if arr.size > 1 else 0.0
    return f"{arr.mean():.4f}+/-{sd:.4f}"


def comparison_cell(full_values, ablation_values):
    """Format one independent-sample one-sided Mann-Whitney comparison."""
    prefix = (f"n={len(full_values)}/{len(ablation_values)}; "
              f"{mean_sd(full_values)} vs {mean_sd(ablation_values)}")
    if len(full_values) == 0 or len(ablation_values) == 0:
        return f"{prefix}; U=NA; p=NA", None
    with warnings.catch_warnings():
        # scipy can warn about ties/exact-method selection; the result remains valid.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = mannwhitneyu(full_values, ablation_values, alternative="greater")
    star = " *" if result.pvalue < 0.05 else ""
    return f"{prefix}; U={result.statistic:.2f}; p={result.pvalue:.4g}{star}", result.pvalue


def build_report(doc):
    results = doc.get("results", {})
    repeats = doc.get("repeats", "?")
    lines = [
        "Ablation: full agent vs single-component ablations (best_agg)",
        "One-sided Mann-Whitney U tests; hypothesis: full's best aggregate is higher.",
        "Observations are completed repeats; values are mean+/-sample SD.",
        "Cell format: n(full/ablation); full mean+/-SD vs ablation mean+/-SD; U; p.",
        "* p < 0.05 (unadjusted one-sided p value).",
        "",
        f"Input: results/ablation.json (repeats={repeats}).",
        "",
    ]

    targets = [t for t in TARGET_ORDER if t in results]
    targets.extend(sorted(set(results).difference(targets)))
    if not targets:
        lines.append("No ablation data loaded.")
        return "\n".join(lines) + "\n"

    significant = []
    for ablation in ABLATIONS:
        lines.append(f"Ablation removed: {ablation}")
        lines.append("=" * (18 + len(ablation)))
        headers = ["Target", "agg"]
        rows = []
        for target in targets:
            full_vals = results.get(target, {}).get("full", {}).get("best_agg", [])
            abl_vals = results.get(target, {}).get(ablation, {}).get("best_agg", [])
            full_vals = [float(v) for v in full_vals if v is not None]
            abl_vals = [float(v) for v in abl_vals if v is not None]
            cell, pvalue = comparison_cell(full_vals, abl_vals)
            rows.append((target, cell))
            if pvalue is not None and pvalue < 0.05:
                significant.append((target, ablation, pvalue))
        widths = [len(h) for h in headers]
        for target, cell in rows:
            widths[0] = max(widths[0], len(target))
            widths[1] = max(widths[1], len(cell))
        lines.append("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        lines.append("  ".join("-" * w for w in widths))
        for target, cell in rows:
            lines.append("  ".join(
                c.ljust(widths[i]) for i, c in enumerate((target, cell))))
        lines.append("")

    lines.append("Full agent significantly superior (unadjusted p < 0.05):")
    if significant:
        lines.extend(
            f"- ({target}, best_agg, {ablation}): p={pvalue:.4g}"
            for target, ablation, pvalue in significant
        )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Ablation JSON path (default: results/ablation.json).",
    )
    parser.add_argument(
        "--out",
        default="results/ablation_significance.txt",
        help="Text report path; use '-' to write only to stdout.",
    )
    args = parser.parse_args()

    path = resolve_path(args.input)
    with path.open() as f:
        doc = json.load(f)
    report = build_report(doc)
    print(report, end="")
    if args.out != "-":
        output = resolve_path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
        print(f"saved -> {output}")


if __name__ == "__main__":
    main()
