"""Orthogonal validation for the smina docking protocol.

Two cheap, local, no-retrain checks that answer the reviewer question "are your
docking scores trustworthy?":

  Task A (reproducibility): re-dock the benchmark's aggregate-top-5 candidates
      under three different smina seeds and check (i) per-molecule score std and
      (ii) Spearman rank correlation between seeds. A single seed=42 docking must
      be enough to support relative method ranking.

  Task B (redocking RMSD): re-dock each target's co-crystal ligand into the same
      box used for the benchmark and measure heavy-atom symmetry-corrected RMSD
      against the deposited crystal pose. RMSD <= 2.0 A = "reproduces crystal
      pose". This validates the box, not the ligand (co-crystal ligands are
      deep hits used only for box validation, never in the paper narrative).

Run:
    python scripts/orthogonal_validation.py reproducibility
    python scripts/orthogonal_validation.py redock
    python scripts/orthogonal_validation.py all
"""
import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scipy.stats import spearmanr                       # noqa: E402
from medchem_agent import config                        # noqa: E402
from medchem_agent.tools import docking                 # noqa: E402

TARGETS = ["EGFR", "CDK2", "ABL1", "BRAF", "VEGFR2", "SRC"]
METHODS = ["agent", "random", "greedy", "llm_blind", "ga", "gated_ga"]
BENCH_FILES = [
    "results/benchmark.json",
    "results/benchmark_new.json",
    "results/benchmark_s345.json",
]
REDOCK_SEEDS = [42, 1234, 9999]
TOP_K = 5
EXHAUSTIVENESS = 8
CPU = 4
WORKERS = 32
RESULT_A = ROOT / "results" / "orthogonal_reproducibility.json"
RESULT_B = ROOT / "results" / "orthogonal_redock_rmsd.json"

# Co-crystal ligand SMILES, taken from the RCSB chemical-component ideal SDF
# (ligand_res in target.json). Only used to validate the box in Task B.
COCRYSTAL_SMILES = {
    "EGFR":   "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",                     # AQ4  erlotinib
    "CDK2":   "CN[C@@H]1C[C@H]2O[C@@](C)([C@@H]1OC)n1c3ccccc3c3c4c(c5c6ccccc6n2c5c31)C(=O)NC4",  # STU  staurosporine
    "ABL1":   "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",    # STI  imatinib
    "BRAF":   "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F",  # 032  vemurafenib
    "VEGFR2": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",      # BAX  sorafenib
    "SRC":    "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",        # 1N1  dasatinib
}


def load_target(name: str) -> dict:
    with open(ROOT / "data" / "targets" / name / "target.json") as f:
        return json.load(f)


# ---------------------------------------------------------------- Task A

def _dock_worker(args):
    """Module-level smina worker (picklable under fork start)."""
    smi, receptor_pdb, center, size, seed = args
    d = docking.dock(smi, receptor_pdb, center=center, size=size,
                     exhaustiveness=EXHAUSTIVENESS, cpu=CPU, seed=seed)
    return smi, seed, d["affinity"]


def collect_pool():
    """Merge the three benchmark files into {target: {seed: {method: [...]}}}."""
    raw = {}
    for f in BENCH_FILES:
        d = json.load(open(ROOT / f))
        for tgt, seeds in d["raw"].items():
            raw.setdefault(tgt, {}).update(seeds)
    return raw


def candidate_pool(raw):
    """Per target, the set of aggregate-top-5 candidate SMILES across 6 seeds x 6
    methods (deduplicated). The redocking set for Task A."""
    pool = {}
    for tgt in TARGETS:
        smis = set()
        for seed, methods in raw[tgt].items():
            for m in METHODS:
                cands = [c for c in methods.get(m, []) if c.get("affinity") is not None]
                cands.sort(key=lambda c: -(c.get("aggregate") or 0))
                for c in cands[:TOP_K]:
                    smis.add(c["smiles"])
        pool[tgt] = sorted(smis)
    return pool


def run_reproducibility():
    raw = collect_pool()
    pool = candidate_pool(raw)
    total = sum(len(v) for v in pool.values())
    print(f"Task A: re-docking {total} unique candidates x {len(REDOCK_SEEDS)} seeds "
          f"= {total * len(REDOCK_SEEDS)} smina runs", flush=True)

    out = {}
    with Pool(WORKERS) as p:
        for tgt in TARGETS:
            meta = load_target(tgt)
            center = tuple(meta["box_center"])
            size = tuple(meta["box_size"])
            receptor = str(config.resolve_path(meta["receptor_pdb"]))
            tasks = [(smi, receptor, center, size, seed)
                     for smi in pool[tgt] for seed in REDOCK_SEEDS]
            results = p.map(_dock_worker, tasks)

            per_smi = defaultdict(dict)
            for smi, seed, aff in results:
                per_smi[smi][f"seed{seed}"] = aff
            out[tgt] = {smi: vals for smi, vals in per_smi.items()}
            print(f"  {tgt}: {len(out[tgt])} molecules done", flush=True)

    # per-molecule std across the three seeds
    for tgt, mols in out.items():
        for smi, vals in mols.items():
            scores = [vals[f"seed{s}"] for s in REDOCK_SEEDS if vals.get(f"seed{s}") is not None]
            vals["std"] = _std(scores) if len(scores) > 1 else None

    RESULT_A.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_A, "w") as f:
        json.dump(out, f, indent=2)

    # summary stats
    rho12, rho13, std_med = summarize_reproducibility(out)
    print(f"\nsaved -> {RESULT_A}")
    print(f"Spearman rho (seed42 vs seed1234) = {rho12:.3f}")
    print(f"Spearman rho (seed42 vs seed9999) = {rho13:.3f}")
    print(f"median per-molecule seed std      = {std_med:.3f} kcal/mol")


def _std(xs):
    if not xs:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def summarize_reproducibility(out):
    a12, b12, a13, c13, stds = [], [], [], [], []
    for tgt, mols in out.items():
        for smi, vals in mols.items():
            a, b, c = vals.get("seed42"), vals.get("seed1234"), vals.get("seed9999")
            if vals.get("std") is not None:
                stds.append(vals["std"])
            if a is not None and b is not None:
                a12.append(a); b12.append(b)
            if a is not None and c is not None:
                a13.append(a); c13.append(c)
    rho12 = spearmanr(a12, b12).statistic if len(a12) > 1 else float("nan")
    rho13 = spearmanr(a13, c13).statistic if len(a13) > 1 else float("nan")
    std_med = sorted(stds)[len(stds) // 2] if stds else float("nan")
    return rho12, rho13, std_med


# ---------------------------------------------------------------- Task B

def _extract_submol(mol, atom_indices):
    """Extract the subgraph over atom_indices, preserving 3D coordinates."""
    from rdkit import Chem
    idx_set = set(atom_indices)
    rw = Chem.RWMol()
    old2new = {}
    for idx in atom_indices:
        a = mol.GetAtomWithIdx(idx)
        old2new[idx] = rw.AddAtom(Chem.Atom(a.GetAtomicNum()))
    for b in mol.GetBonds():
        if b.GetBeginAtomIdx() in idx_set and b.GetEndAtomIdx() in idx_set:
            rw.AddBond(old2new[b.GetBeginAtomIdx()], old2new[b.GetEndAtomIdx()],
                       b.GetBondType())
    conf = mol.GetConformer()
    newconf = Chem.Conformer(len(atom_indices))
    for old, new in old2new.items():
        newconf.SetAtomPosition(new, conf.GetAtomPosition(old))
    m = rw.GetMol()
    m.AddConformer(newconf, assignId=True)
    return m


def native_pose(name):
    """Co-crystal ligand pose from the deposited PDB, bond orders from the ideal
    SMILES template. Returns an RDKit mol with 3D coords (heavy atoms)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    meta = load_target(name)
    pdb = ROOT / "data" / "pdb" / f"{meta['pdb_id']}.pdb"
    res, chain = meta["ligand_res"], meta["chain"]
    pdb_mol = Chem.MolFromPDBFile(str(pdb), sanitize=False, removeHs=False)
    lig_idx = [a.GetIdx() for a in pdb_mol.GetAtoms()
               if a.GetPDBResidueInfo() is not None
               and a.GetPDBResidueInfo().GetResidueName().strip() == res
               and a.GetPDBResidueInfo().GetChainId() == chain
               and a.GetPDBResidueInfo().GetAltLoc().strip() in ("", "A")]
    if not lig_idx:
        return None
    native = _extract_submol(pdb_mol, lig_idx)
    template = Chem.MolFromSmiles(COCRYSTAL_SMILES[name])
    return AllChem.AssignBondOrdersFromTemplate(template, native)


def best_pose(sdf_path):
    """Lowest-affinity pose from a multi-pose smina output SDF (matches
    docking._read_best_score's property order)."""
    from rdkit import Chem
    best, best_score = None, None
    with Chem.rdBase.BlockLogs():
        for m in Chem.SDMolSupplier(sdf_path):
            if m is None:
                continue
            for prop in ("minimizedAffinity", "CNNaffinity", "CNNscore"):
                if m.HasProp(prop):
                    v = float(m.GetProp(prop))
                    if best_score is None or v < best_score:
                        best, best_score = m, v
                    break
    return best, best_score


def run_redock():
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign

    out = {}
    print("Task B: co-crystal ligand redocking RMSD (<=2.0 A = reproduced)", flush=True)
    for name in TARGETS:
        meta = load_target(name)
        smi = COCRYSTAL_SMILES[name]
        native = native_pose(name)
        if native is None:
            out[name] = {"rmsd": None, "error": f"no ligand {meta['ligand_res']} found"}
            print(f"  {name}: ERROR no native ligand", flush=True)
            continue

        d = docking.dock(smi, str(config.resolve_path(meta["receptor_pdb"])),
                         center=tuple(meta["box_center"]), size=tuple(meta["box_size"]),
                         exhaustiveness=EXHAUSTIVENESS, cpu=CPU, seed=42, cleanup=False)
        if d["affinity"] is None or not d["workdir"]:
            out[name] = {"rmsd": None, "error": d.get("stderr") or "dock failed"}
            print(f"  {name}: dock failed {d.get('stderr')}", flush=True)
            continue

        pose, score = best_pose(Path(d["workdir"]) / "docked.sdf")
        docked = Chem.RemoveHs(pose)
        try:
            rmsd = rdMolAlign.GetBestRMS(docked, native)
        except Exception as e:  # noqa: BLE001 - fall back to explicit atom map
            match = docked.GetSubstructMatch(native)
            if match:
                rmsd = rdMolAlign.CalcRMS(docked, native, list(enumerate(match)))
            else:
                rmsd = float("nan")

        out[name] = {
            "pdb": meta["pdb_id"],
            "ligand_res": meta["ligand_res"],
            "ligand": COCRYSTAL_SMILES[name],
            "redock_affinity": score,
            "rmsd": round(float(rmsd), 3),
            "reproduced": bool(rmsd <= 2.0),
        }
        print(f"  {name} ({meta['pdb_id']}/{meta['ligand_res']}): RMSD = {rmsd:.3f} A "
              f"(affinity {score:.2f})", flush=True)

    with open(RESULT_B, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {RESULT_B}")
    print("\n| target | PDB | ligand | RMSD (A) | reproduced |")
    print("|---|---|---|---|---|")
    for name in TARGETS:
        r = out.get(name, {})
        if r.get("rmsd") is None:
            print(f"| {name} | {r.get('pdb','')} | {r.get('ligand_res','')} | n/a | {r.get('error','')} |")
        else:
            print(f"| {name} | {r['pdb']} | {r['ligand_res']} | {r['rmsd']} | "
                  f"{'yes' if r['reproduced'] else 'no'} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", default="all",
                    choices=["reproducibility", "redock", "all"])
    args = ap.parse_args()
    if args.task in ("reproducibility", "all"):
        run_reproducibility()
    if args.task in ("redock", "all"):
        run_redock()


if __name__ == "__main__":
    main()
