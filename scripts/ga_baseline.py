"""Evolutionary (mu + lambda) baseline using the same Scorer.

Maintains a population selected by aggregate score -- strictly stronger than the
greedy hill-climb (which collapses to a single point). This gives the agent a
non-trivial optimization baseline to beat on docking-primary / hypervolume /
diversity.
"""
import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from baseline import load_target, mutate          # noqa: E402
from medchem_agent.evaluate import Scorer         # noqa: E402


def ga_baseline(scorer: Scorer, hit: str, budget: int, pop_size: int = 8,
                seed: int = 0) -> list:
    """(mu+lambda)-ES: evolve a population, select by aggregate, dock <= budget."""
    rng = random.Random(seed)
    # initialize population: the hit + (pop_size-1) mutants
    pop = [hit]
    seen = {hit}
    while len(pop) < pop_size:
        s = mutate(hit, rng)
        if s and s not in seen:
            seen.add(s)
            pop.append(s)
    cands = scorer.score_many(pop)
    if not cands:
        return []

    # evolve until the docking budget is exhausted
    while len(cands) < budget:
        children = []
        for s in pop:
            for _ in range(3):  # retries for invalid/repeat mutations
                c = mutate(s, rng)
                if c and c not in seen:
                    seen.add(c)
                    children.append(c)
                    break
        if not children:
            break
        take = min(len(children), budget - len(cands))
        new = scorer.score_many(children[:take])
        if not new:
            break
        for p in new:
            p.setdefault("origin", "ga")
        cands.extend(new)
        # elitist selection of the next generation
        ranked = sorted(cands, key=lambda c: -(c.get("aggregate") or 0))
        pop = [c["smiles"] for c in ranked[:pop_size]]

    cands.sort(key=lambda c: -(c.get("aggregate") or 0))
    return cands


def gated_ga_baseline(scorer: Scorer, hit: str, budget: int, pop_size: int = 8,
                      qed_min: float = 0.5, sa_max: float = 4.0, seed: int = 0) -> list:
    """(mu+lambda)-ES with docking fitness and a drug-likeness gate.

    The population only admits drug-like candidates (QED>=qed_min, SA<=sa_max)
    and is selected by docking depth, so the search maximizes affinity while
    staying inside the drug-like region -- where neither the LLM (prior-bound)
    nor unconstrained search (drifts non-drug-like) reaches.
    """
    rng = random.Random(seed)

    def passes(c):
        return (c.get("qed") or 0.0) >= qed_min and (c.get("sa") or 99.0) <= sa_max

    pop = [hit]
    seen = {hit}
    while len(pop) < pop_size:
        s = mutate(hit, rng)
        if s and s not in seen:
            seen.add(s)
            pop.append(s)
    cands = scorer.score_many(pop)

    while len(cands) < budget:
        children = []
        for s in pop:
            for _ in range(3):
                c = mutate(s, rng)
                if c and c not in seen:
                    seen.add(c)
                    children.append(c)
                    break
        if not children:
            break
        take = min(len(children), budget - len(cands))
        cands.extend(scorer.score_many(children[:take]))
        druglike = [c for c in cands if c.get("affinity") is not None and passes(c)]
        if druglike:
            ranked = sorted(druglike, key=lambda c: c.get("affinity") or 0.0)
            pop = [c["smiles"] for c in ranked[:pop_size]]

    cands.sort(key=lambda c: c.get("affinity") or 0.0)
    return cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--proposals", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop-size", type=int, default=8)
    ap.add_argument("--gated", action="store_true",
                    help="docking fitness + drug-like gate (gated_ga_baseline)")
    args = ap.parse_args()

    meta = load_target(args.target)
    hit = meta["hit_smiles"]
    budget = args.rounds * args.proposals
    scorer = Scorer(meta)
    if args.gated:
        cands = gated_ga_baseline(scorer, hit, budget, args.pop_size, args.seed)
        gated = [c for c in cands
                 if (c.get("qed") or 0.0) >= 0.5 and (c.get("sa") or 99.0) <= 4.0]
        best = min(gated, key=lambda c: c["affinity"]) if gated else None
        if best:
            print(f"[gated_ga] n={len(cands)} best_druglike_dock={best['affinity']} "
                  f"qed={best.get('qed'):.2f} sa={best.get('sa'):.2f}")
    else:
        cands = ga_baseline(scorer, hit, budget, args.pop_size, args.seed)
        if cands:
            print(f"[ga] budget={budget} n={len(cands)} "
                  f"best_agg={cands[0]['aggregate']:.3f} dock={cands[0]['affinity']}")
    scorer.close()


if __name__ == "__main__":
    main()
