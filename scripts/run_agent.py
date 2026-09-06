"""Run MedChemAgent on one target and save results."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medchem_agent import scoring  # noqa: E402
from medchem_agent.agent import MedChemAgent  # noqa: E402


def load_target(name: str) -> dict:
    path = ROOT / "data" / "targets" / name / "target.json"
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--proposals", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--exhaustiveness", type=int, default=None)
    ap.add_argument("--no-rag", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # runtime overrides (leave config.yaml untouched for reproducibility record)
    import medchem_agent.config as cfg
    if args.workers is not None:
        cfg.set("agent", "workers", args.workers)
    if args.exhaustiveness is not None:
        cfg.set("agent", "exhaustiveness", args.exhaustiveness)

    meta = load_target(args.target)
    agent = MedChemAgent(meta, use_rag=not args.no_rag, verbose=True)
    cands = agent.run(meta["hit_smiles"],
                      max_rounds=args.rounds,
                      proposals_per_round=args.proposals)

    front = scoring.pareto_front(cands)

    print(f"\n=== {args.target}: {len(cands)} candidates, {len(front)} on Pareto front ===")
    print("\nTop 5 by aggregate score:")
    for c in cands[:5]:
        print(f"  {c['smiles'][:60]:60s} agg={c['aggregate']:.2f} "
              f"dock={c['affinity']} qed={c['qed']:.2f} sa={c['sa']:.2f} "
              f"admet={c['admet_pass']}/{c['admet_total']} (round {c['round']})")

    out = args.out or str(ROOT / "results" / f"{args.target}_agent.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"target": args.target, "candidates": cands, "pareto": front}, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
