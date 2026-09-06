"""Head-to-head benchmark: Pareto agent vs scalar agent vs GA vs baselines.

Reports four metrics per method -- docking-primary (drug-like gate), aggregate,
Pareto hypervolume, candidate diversity -- the metrics that matter for the
method contribution. Runs on whatever hit_smiles the target.json files currently
specify (so set the weak hits there first).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from medchem_agent import pareto                                  # noqa: E402
from medchem_agent.agent import MedChemAgent                      # noqa: E402
from medchem_agent.evaluate import Scorer                         # noqa: E402
from baseline import (                                            # noqa: E402
    greedy_baseline, llm_blind_baseline, load_target, random_baseline,
)
from ga_baseline import ga_baseline, gated_ga_baseline            # noqa: E402

TARGETS = ["EGFR", "CDK2", "ABL1"]
METHODS = ["agent", "random", "greedy", "llm_blind", "ga", "gated_ga"]


def metrics(cands):
    """Summarize a method's candidate set into the four headline metrics."""
    scored = [c for c in cands if c.get("affinity") is not None]
    if not scored:
        return None
    agg_best = max(scored, key=lambda c: c.get("aggregate") or 0)
    gated = pareto.docking_best_gated(scored)
    front = pareto.pareto_front_raw(scored)
    return {
        "n": len(scored),
        "agg": round(agg_best.get("aggregate") or 0, 4),
        "agg_dock": round(agg_best.get("affinity"), 3),
        "dock_gated": round(gated["affinity"], 3) if gated else None,
        "dock_gated_dl": round(pareto.druglikeness(gated), 3) if gated else None,
        "hypervolume": round(pareto.hypervolume(front), 4),
        "diversity": round(pareto.diversity(scored), 4),
    }


def run_target(meta, hit, rounds, per_round, seed):
    budget = rounds * per_round
    res = {}

    print("  [agent] running...", flush=True)
    ag = MedChemAgent(meta, use_rag=True, verbose=True)
    res["agent"] = ag.run(hit, max_rounds=rounds, proposals_per_round=per_round)

    scorer = Scorer(meta)
    res["hit"] = [scorer.score_one(hit)]
    print("  [random/greedy/llm_blind/ga/gated_ga] running...", flush=True)
    res["random"] = random_baseline(scorer, hit, budget, seed)
    res["greedy"] = greedy_baseline(scorer, hit, rounds, per_round, seed)
    res["llm_blind"] = llm_blind_baseline(scorer, meta, hit, budget, seed)
    res["ga"] = ga_baseline(scorer, hit, budget, seed=seed)
    res["gated_ga"] = gated_ga_baseline(scorer, hit, budget, seed=seed)
    scorer.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--proposals", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = {}
    metric = {}
    for name in args.targets:
        meta = load_target(name)
        hit = meta["hit_smiles"]
        print(f"\n{'='*60}\n{name}  (hit dock = see hit row)\n{'='*60}", flush=True)
        raw[name] = {}
        metric[name] = {}
        for seed in args.seeds:
            print(f"\n--- seed {seed} ---", flush=True)
            res = run_target(meta, hit, args.rounds, args.proposals, seed)
            raw[name][str(seed)] = res
            metric[name][str(seed)] = {m: metrics(res[m]) for m in res}
            print(f"\n  {'method':10s} {'n':>3s} {'agg':>7s} {'dock_gated':>11s} "
                  f"{'hv':>8s} {'div':>6s}", flush=True)
            for m in METHODS + ["hit"]:
                mt = metric[name][str(seed)].get(m)
                if mt is None:
                    continue
                dg = f"{mt['dock_gated']}" if mt["dock_gated"] is not None else "None"
                print(f"  {m:10s} {mt['n']:3d} {mt['agg']:7.3f} {dg:>11s} "
                      f"{mt['hypervolume']:8.3f} {mt['diversity']:6.3f}", flush=True)

    out = Path(args.out) if args.out else ROOT / "results" / "benchmark.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"rounds": args.rounds, "proposals": args.proposals,
                   "seeds": args.seeds, "raw": raw, "metrics": metric}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
