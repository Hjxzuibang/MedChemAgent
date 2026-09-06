"""Ablation study: full agent vs single-component ablations, multi-repeat.

Removes one component at a time while keeping the same evaluation budget and
Scorer, so any drop in the best aggregate is attributable to the removed
component:

  full        -- RAG + tools + iterative feedback   (the proposed method)
  no_rag      -- tools + feedback, no literature    (use_rag=False)
  no_tools    -- RAG + feedback, no function calls  (use_tools=False)
  no_feedback -- RAG + tools, single-pass (no scored candidates fed back)

Each (target, variant) is run `--repeats` times independently. MedChemAgent has
no seed parameter (LLM temperature 0.7 is the only source of randomness), so a
"repeat" is simply an independent agent.run(...) call -- no RNG plumbing, same
meaning as the benchmark's agent/llm_blind "seed".

Saves results/ablation.json and prints a per-(target, variant) best_agg
mean+/-SD table plus Delta = full_mean - variant_mean (the attribution numbers
for Fig 5 / Table 3). Statistical tests are in run_ablation_significance.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import httpx
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from medchem_agent.agent import MedChemAgent  # noqa: E402
from baseline import load_target  # noqa: E402

TARGETS = ["EGFR", "CDK2", "ABL1", "BRAF", "VEGFR2", "SRC"]

VARIANTS = {
    "full":        dict(use_rag=True,  use_tools=True,  feedback=True),
    "no_rag":      dict(use_rag=False, use_tools=True,  feedback=True),
    "no_tools":    dict(use_rag=True,  use_tools=False, feedback=True),
    "no_feedback": dict(use_rag=True,  use_tools=True,  feedback=False),
}


def best_aggregate(cands):
    """Best aggregate scalar across a run's scored candidates (None if empty)."""
    values = [c.get("aggregate") for c in cands
              if c.get("aggregate") is not None and np.isfinite(c.get("aggregate"))]
    return float(max(values)) if values else None


def mean_sd(values):
    """Return (mean+/-sd string, finite count) for a list possibly holding None."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return "NA", 0
    sd = arr.std(ddof=1) if arr.size > 1 else 0.0
    return f"{arr.mean():.4f}+/-{sd:.4f}", arr.size


TRANSIENT = (
    InternalServerError,   # gateway 5xx
    RateLimitError,        # 429
    APIConnectionError,    # connect/network (parent of APITimeoutError)
    APITimeoutError,       # request timeout
    httpx.HTTPError,       # httpx-level network errors
    TimeoutError,          # built-in socket timeout
)


def run_one(meta, hit, variant, rounds, proposals, max_attempts=30, max_delay=600):
    """One independent agent run, retrying transient LLM-gateway failures.

    The LLM endpoint intermittently returns 5xx under load; llm.chat already
    backoff-retries ~1 min per call,
    but a sustained outage can last tens of minutes. Retrying the whole repeat
    with capped exponential backoff (60s -> 10min, up to ~5h of outage) keeps a
    multi-hour ablation from dying on one such hiccup. Non-transient exceptions
    (real bugs) are re-raised immediately so they don't silently spin.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            agent = MedChemAgent(meta, verbose=True, **VARIANTS[variant])
            cands = agent.run(hit, max_rounds=rounds, proposals_per_round=proposals)
            return best_aggregate(cands), cands
        except Exception as e:  # noqa: BLE001 - transient gateway error on a long job
            if not isinstance(e, TRANSIENT) or attempt >= max_attempts:
                raise
            delay = min(60 * 2 ** (attempt - 1), max_delay)
            print(f"    [run_one] attempt {attempt}/{max_attempts} failed "
                  f"({type(e).__name__}: {str(e)[:120]}) — retrying in {delay}s",
                  flush=True)
            time.sleep(delay)


def save(results, out, args):
    """Write the current (possibly partial) results to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"rounds": args.rounds, "proposals": args.proposals,
                   "repeats": args.repeats, "budget": args.rounds * args.proposals,
                   "results": results}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--proposals", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Reuse completed/partial (target, variant) entries from --out.")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "results" / "ablation.json"
    if not out.is_absolute():
        out = ROOT / out

    budget = args.rounds * args.proposals
    results, start_rep = {}, {}
    if args.resume and out.exists():
        try:
            prev = json.loads(out.read_text()).get("results", {})
        except (OSError, json.JSONDecodeError):
            prev = {}
        n_reused = 0
        for name in args.targets:
            for variant in args.variants:
                entry = prev.get(name, {}).get(variant)
                ba = entry.get("best_agg") if isinstance(entry, dict) else None
                ca = entry.get("cands") if isinstance(entry, dict) else None
                if (isinstance(ba, list) and isinstance(ca, list)
                        and len(ba) == len(ca)):
                    n = min(len(ba), args.repeats)
                    results.setdefault(name, {})[variant] = {
                        "best_agg": ba[:n], "cands": ca[:n]}
                    start_rep[(name, variant)] = n
                    if n > 0:
                        n_reused += 1
                else:
                    start_rep[(name, variant)] = 0
        if n_reused:
            print(f"[resume] reusing {n_reused} completed/partial "
                  f"(target, variant) entries", flush=True)

    for name in args.targets:
        meta = load_target(name)
        hit = meta["hit_smiles"]
        results.setdefault(name, {})
        print(f"\n{'='*60}\n{name} (budget={budget}, repeats={args.repeats})\n"
              f"{'='*60}", flush=True)
        for variant in args.variants:
            s = start_rep.get((name, variant), 0)
            if s >= args.repeats:
                print(f"[{variant}] already complete (resumed), skipping", flush=True)
                continue
            entry = results[name].get(variant) or {}
            best_list = list(entry.get("best_agg", []))
            cands_list = list(entry.get("cands", []))
            if s > 0:
                print(f"[{variant}] resuming from repeat {s + 1}/{args.repeats} "
                      f"({s} done)", flush=True)
            for rep in range(s, args.repeats):
                print(f"[{variant}] repeat {rep + 1}/{args.repeats} ...", flush=True)
                b, cands = run_one(meta, hit, variant, args.rounds, args.proposals)
                best_list.append(b)
                cands_list.append(cands)
                tag = f"{b:.4f}" if b is not None else "NA"
                print(f"    best_agg={tag}  n_cands={len(cands)}", flush=True)
                results[name][variant] = {"best_agg": best_list, "cands": cands_list}
                save(results, out, args)  # checkpoint every repeat (survives crash)

    # --- summary table: mean+/-SD and Delta = full_mean - variant_mean ---
    print("\n" + "=" * 72, flush=True)
    print("best_agg mean+/-SD per (target, variant); Delta = full_mean - variant_mean",
          flush=True)
    print("=" * 72, flush=True)
    for name in args.targets:
        full_arr = np.asarray(
            [v for v in results[name]["full"]["best_agg"] if v is not None], dtype=float)
        full_mean = float(full_arr.mean()) if full_arr.size else None
        print(f"\n{name}", flush=True)
        header = f"  {'variant':12s} {'n':>2s} {'best_agg mean+/-SD':>22s} {'Delta':>8s}"
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)
        for variant in args.variants:
            vals = results[name][variant]["best_agg"]
            label, n = mean_sd(vals)
            arr = np.asarray([v for v in vals if v is not None], dtype=float)
            if variant == "full":
                delta = ""
            elif full_mean is not None and arr.size:
                delta = f"{full_mean - arr.mean():+.4f}"
            else:
                delta = "NA"
            print(f"  {variant:12s} {n:2d} {label:>22s} {delta:>8s}", flush=True)

    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
