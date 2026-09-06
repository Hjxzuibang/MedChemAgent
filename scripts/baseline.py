"""Baseline methods for closed-loop hit-to-lead optimization.

Compares the full MedChemAgent against three baselines that isolate the value of
(a) chemical knowledge and (b) the iterative feedback loop:

  random     -- random single-point analogs of the hit, no feedback
  greedy     -- iterative hill-climbing (mutate -> keep best), feedback but no chemistry
  llm_blind  -- LLM proposes analogs in one shot, no scoring feedback / no tools
  agent      -- LLM + tools + iterative feedback (the proposed method)

Every method scores the same number of molecules with the same Scorer, so any
improvement is attributable to the method itself, not to a larger budget.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from medchem_agent import llm, scoring  # noqa: E402
from medchem_agent.evaluate import Scorer  # noqa: E402

# one-point substituents ('*' = attachment point). Monovalent, drug-like.
FRAGMENTS = ["*F", "*Cl", "*Br", "*C#N", "*C(F)(F)F", "*O", "*N", "*OC", "*C", "*C1CC1"]

LLM_BLIND_SYSTEM = (
    "You are a senior medicinal chemist. Given a protein target and a starting hit, "
    "propose chemically valid analogs that improve drug-likeness and binding. "
    "Return ONLY a JSON object: {\"proposals\": [{\"smiles\": \"...\", \"rationale\": \"...\"}]}."
)


def load_target(name: str) -> dict:
    with open(ROOT / "data" / "targets" / name / "target.json") as f:
        return json.load(f)


# --------------------------------------------------------------------- #
# Random mutation operator (shared by random / greedy baselines)
# --------------------------------------------------------------------- #
def _attach_fragment(rw: Chem.RWMol, parent_idx: int, frag_smiles: str) -> bool:
    """Attach `frag_smiles` (with a dummy '*' atom) to `parent_idx`. Mutates rw."""
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return False
    frag = Chem.AddHs(frag)
    dummy = next((a for a in frag.GetAtoms() if a.GetSymbol() == "*"), None)
    if dummy is None:
        return False

    idx = {}
    for a in frag.GetAtoms():
        if a.GetIdx() == dummy.GetIdx():
            continue
        idx[a.GetIdx()] = rw.AddAtom(a)
    for b in frag.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i == dummy.GetIdx() or j == dummy.GetIdx():
            continue
        rw.AddBond(idx[i], idx[j], b.GetBondType())

    attach = dummy.GetNeighbors()[0].GetIdx()
    rw.AddBond(parent_idx, idx[attach], Chem.BondType.SINGLE)
    return True


def mutate(smi: str, rng: random.Random):
    """Randomly substitute one H (on a C bearing H) with a small fragment."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    rw = Chem.RWMol(Chem.AddHs(m))
    # after AddHs, hydrogens are explicit neighbor atoms (GetNumExplicitHs is 0),
    # so select carbons that actually carry an H neighbor.
    carbons = [a.GetIdx() for a in rw.GetAtoms()
               if a.GetSymbol() == "C" and any(n.GetSymbol() == "H" for n in a.GetNeighbors())]
    if not carbons:
        return None
    c_idx = rng.choice(carbons)
    hs = [n.GetIdx() for n in rw.GetAtomWithIdx(c_idx).GetNeighbors() if n.GetSymbol() == "H"]
    if not hs:
        return None
    rw.RemoveAtom(hs[0])
    if not _attach_fragment(rw, c_idx, rng.choice(FRAGMENTS)):
        return None
    try:
        m2 = rw.GetMol()
        Chem.SanitizeMol(m2)
    except Exception:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(m2))


# --------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------- #
def random_baseline(scorer: Scorer, hit: str, n: int, seed: int = 0) -> list:
    """n independent single-point mutants of the hit; report all (sorted)."""
    rng = random.Random(seed)
    smis, seen = [], {hit}
    for _ in range(n * 4):  # oversample: many mutations are invalid/repeats
        s = mutate(hit, rng)
        if s and s not in seen:
            seen.add(s)
            smis.append(s)
        if len(smis) >= n:
            break
    cands = scorer.score_many(smis)
    cands.sort(key=lambda x: -(x.get("aggregate") or 0))
    return cands


def greedy_baseline(scorer: Scorer, hit: str, rounds: int, per_round: int, seed: int = 0) -> list:
    """Hill-climb: each round mutate the current best, keep the top aggregate."""
    rng = random.Random(seed)
    best = scorer.score_one(hit)
    best["round"] = 0
    best["origin"] = "greedy"
    all_cands = [best]
    cur = hit
    for r in range(1, rounds + 1):
        seen = {c["smiles"] for c in all_cands}
        mutants, guard = [], 0
        while len(mutants) < per_round and guard < per_round * 10:
            guard += 1
            s = mutate(cur, rng)
            if s and s not in seen:
                seen.add(s)
                mutants.append(s)
        for p in scorer.score_many(mutants):
            p["round"] = r
            p["origin"] = "greedy"
            all_cands.append(p)
        # keep the single best as the new parent for the next round
        cur = max(all_cands, key=lambda x: x.get("aggregate") or 0)["smiles"]
    all_cands.sort(key=lambda x: -(x.get("aggregate") or 0))
    return all_cands


def llm_blind_propose(target_meta: dict, hit: str, n: int) -> list:
    """LLM proposes n analogs in one shot, no tools, no feedback."""
    user = (f"Target: {target_meta['name']} (co-crystal ligand: {target_meta['drug']}).\n"
            f"Starting hit SMILES: {hit}\n"
            f"Propose {n} chemically valid analogs and return them as JSON.")
    resp = llm.chat([
        {"role": "system", "content": LLM_BLIND_SYSTEM},
        {"role": "user", "content": user},
    ])
    text = resp.choices[0].message.content or ""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    return [p.get("smiles", "") for p in data.get("proposals", [])][:n]


def llm_blind_baseline(scorer: Scorer, target_meta: dict, hit: str, n: int, seed: int = 0) -> list:
    smis = llm_blind_propose(target_meta, hit, n)
    cands = scorer.score_many(smis)
    cands.sort(key=lambda x: -(x.get("aggregate") or 0))
    return cands


# --------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--proposals", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = load_target(args.target)
    hit = meta["hit_smiles"]
    budget = args.rounds * args.proposals
    scorer = Scorer(meta)

    print(f"=== {args.target} baselines (budget={budget} docked mols each) ===", flush=True)
    start = scorer.score_one(hit)
    print(f"hit: dock={start['affinity']} qed={start['qed']:.2f} "
          f"sa={start['sa']:.2f} agg={start['aggregate']:.2f}", flush=True)

    random_cands = random_baseline(scorer, hit, budget, args.seed)
    print(f"[random] best agg={random_cands[0]['aggregate']:.3f} "
          f"dock={random_cands[0]['affinity']}", flush=True)

    greedy_cands = greedy_baseline(scorer, hit, args.rounds, args.proposals, args.seed)
    print(f"[greedy] best agg={greedy_cands[0]['aggregate']:.3f} "
          f"dock={greedy_cands[0]['affinity']}", flush=True)

    llm_cands = llm_blind_baseline(scorer, meta, hit, budget, args.seed)
    print(f"[llm_blind] best agg={llm_cands[0]['aggregate']:.3f} "
          f"dock={llm_cands[0]['affinity']}", flush=True)

    scorer.close()

    result = {
        "target": args.target,
        "budget": budget,
        "seed": args.seed,
        "start": start,
        "random": random_cands,
        "greedy": greedy_cands,
        "llm_blind": llm_cands,
    }
    out = args.out or str(ROOT / "results" / f"{args.target}_baselines.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
