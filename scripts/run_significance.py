"""Compare the Pareto agent against each benchmark baseline by target.

Each observation is one completed seed.  The script intentionally compares
seed-level summary metrics, never the individual molecules within a seed.
"""
import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("random", "greedy", "llm_blind", "ga", "gated_ga")
TARGET_ORDER = ("EGFR", "CDK2", "ABL1", "BRAF", "VEGFR2", "SRC")
METRICS = (
    ("agg", "greater"),
    ("hypervolume", "greater"),
    ("diversity", "greater"),
    ("dock_gated", "less"),
)
DEFAULT_INPUTS = (
    "results/benchmark.json",
    "results/benchmark_new.json",
    "results/benchmark_s345.json",
)


def resolve_path(path_string):
    """Resolve relative paths from the repository root, not the caller's CWD."""
    path = Path(path_string)
    return path if path.is_absolute() else ROOT / path


def load_metrics(input_paths):
    """Load unique target/seed records, tolerating absent or partial result files."""
    records = defaultdict(dict)
    notes = []
    for path_string in input_paths:
        path = resolve_path(path_string)
        if not path.exists():
            notes.append(f"warning: missing input skipped: {path}")
            continue
        try:
            with path.open() as handle:
                document = json.load(handle)
        except (OSError, json.JSONDecodeError) as error:
            notes.append(f"warning: unreadable JSON skipped: {path} ({error})")
            continue

        metrics = document.get("metrics")
        if not isinstance(metrics, dict):
            notes.append(f"warning: no top-level metrics mapping: {path}")
            continue
        notes.append(f"loaded: {path}")
        for target, seed_records in metrics.items():
            if not isinstance(seed_records, dict):
                notes.append(f"warning: invalid seed mapping skipped: {target} in {path}")
                continue
            for seed, method_records in seed_records.items():
                seed = str(seed)
                if seed in records[target]:
                    notes.append(
                        f"warning: duplicate {target}/seed={seed} in {path}; first record retained"
                    )
                    continue
                if isinstance(method_records, dict):
                    records[target][seed] = method_records
                else:
                    notes.append(f"warning: invalid method mapping skipped: {target}/seed={seed}")
    return records, notes


def values_for(records, target, method, metric):
    """Return finite per-seed values; absent methods/metrics are deliberately omitted."""
    values = []
    for seed in sorted(records[target], key=lambda item: (not item.isdigit(), item)):
        entry = records[target][seed].get(method)
        if not isinstance(entry, dict):
            continue
        value = entry.get(metric)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def mean_sd(values):
    if len(values) == 0:
        return "NA"
    sd = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.4f}+/-{sd:.4f}"


def comparison_cell(agent_values, baseline_values, alternative):
    """Format one independent-sample one-sided Mann-Whitney comparison."""
    prefix = (
        f"n={len(agent_values)}/{len(baseline_values)}; "
        f"{mean_sd(agent_values)} vs {mean_sd(baseline_values)}"
    )
    if len(agent_values) == 0 or len(baseline_values) == 0:
        return f"{prefix}; U=NA; p=NA", None
    with warnings.catch_warnings():
        # scipy can warn about ties/exact-method selection; the calculation remains valid.
        warnings.simplefilter("ignore", RuntimeWarning)
        result = mannwhitneyu(agent_values, baseline_values, alternative=alternative)
    star = " *" if result.pvalue < 0.05 else ""
    return f"{prefix}; U={result.statistic:.2f}; p={result.pvalue:.4g}{star}", result.pvalue


def build_report(records, notes):
    lines = [
        "Agent vs baseline: one-sided Mann-Whitney U tests",
        "Observations are completed seeds; values are mean+/-sample SD.",
        "Directions: agg/hypervolume/diversity agent > baseline; dock_gated agent < baseline.",
        "Cell format: n(agent/baseline); agent mean+/-SD vs baseline mean+/-SD; U; p.",
        "* p < 0.05 (unadjusted one-sided p value).",
        "",
        "Inputs and completeness:",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.append("")

    targets = [target for target in TARGET_ORDER if target in records]
    targets.extend(sorted(set(records).difference(targets)))
    if not targets:
        lines.append("No usable metrics were loaded.")
        return "\n".join(lines) + "\n"

    lines.append("Completed seed records by target: " + ", ".join(
        f"{target}=n{len(records[target])} ({','.join(sorted(records[target]))})"
        for target in targets
    ))
    lines.append("")

    significant = []
    for baseline in BASELINES:
        lines.extend((f"Baseline: {baseline}", "=" * (10 + len(baseline))))
        headers = ["Target"] + [metric for metric, _ in METRICS]
        rows = []
        for target in targets:
            row = [target]
            for metric, alternative in METRICS:
                agent_values = values_for(records, target, "agent", metric)
                baseline_values = values_for(records, target, baseline, metric)
                cell, pvalue = comparison_cell(agent_values, baseline_values, alternative)
                row.append(cell)
                if pvalue is not None and pvalue < 0.05:
                    significant.append((target, metric, baseline, pvalue))
            rows.append(row)
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))
        lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
        lines.append("  ".join("-" * width for width in widths))
        lines.extend("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows)
        lines.append("")

    lines.append("Agent significantly superior (unadjusted p < 0.05):")
    if significant:
        lines.extend(
            f"- ({target}, {metric}, {baseline}): p={pvalue:.4g}"
            for target, metric, baseline, pvalue in significant
        )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Benchmark JSON paths. Missing/incomplete files are skipped with a warning.",
    )
    parser.add_argument(
        "--out",
        default="results/benchmark_significance.txt",
        help="Text report path; use '-' to write only to stdout.",
    )
    args = parser.parse_args()

    records, notes = load_metrics(args.inputs)
    report = build_report(records, notes)
    print(report, end="")
    if args.out != "-":
        output = resolve_path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report)
        print(f"saved -> {output}")


if __name__ == "__main__":
    main()
