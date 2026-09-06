"""Shared candidate evaluation: parallel multi-objective scoring.

Both the agent loop and the baseline methods (random / greedy / LLM-blind)
score candidates the same way -- fast RDKit+ADMET properties, then smina
docking (parallelized across molecules), then an aggregate scalar. Keeping
this in one place makes the benchmark comparisons like-for-like.
"""
import math
import multiprocessing

from . import config, scoring
from .tools import admet, docking, rdkit_tools


def fast_score(smi: str):
    """RDKit drug-likeness + ADMET filters (no docking). None if SMILES invalid."""
    p = rdkit_tools.score_molecule(smi)
    if p is None:
        return None
    if not (math.isfinite(p["qed"]) and math.isfinite(p["sa"])):
        return None  # unkekulizable / pathological structure
    ap, at = admet.score_smiles(smi)
    p["admet_pass"], p["admet_total"] = ap, at
    return p


def _dock_one(args):
    """Module-level worker for parallel docking (picklable under fork start)."""
    smi, receptor_pdb, center, size, exhaustiveness, smina_cpu = args
    d = docking.dock(smi, receptor_pdb, center=center, size=size,
                     exhaustiveness=exhaustiveness, cpu=smina_cpu)
    return d.get("affinity")


class Scorer:
    """Evaluates candidates for one target, docking in parallel across molecules."""

    def __init__(self, target_meta: dict, exhaustiveness: int = None,
                 smina_cpu: int = None, workers: int = None):
        self.receptor_pdb = str(config.resolve_path(target_meta["receptor_pdb"]))
        self.center = tuple(target_meta["box_center"])
        self.size = tuple(target_meta["box_size"])
        self.exhaustiveness = exhaustiveness or config.get("agent", "exhaustiveness", default=8)
        self.smina_cpu = smina_cpu or config.get("agent", "smina_cpu", default=4)
        self.workers = workers or config.get("agent", "workers", default=32)
        self._pool = None

    def _ensure_pool(self):
        if self._pool is None and self.workers > 1:
            self._pool = multiprocessing.Pool(self.workers)

    def score_one(self, smi: str):
        """Full score of a single molecule (fast + docking + aggregate)."""
        p = fast_score(smi)
        if p is None:
            return None
        aff = docking.dock(smi, self.receptor_pdb, center=self.center, size=self.size,
                           exhaustiveness=self.exhaustiveness, cpu=self.smina_cpu)["affinity"]
        p["affinity"] = aff
        p["aggregate"] = scoring.aggregate(p)
        return p

    def score_many(self, smiles: list):
        """Full score of many molecules (docking parallelized). Drops invalid SMILES."""
        props = [(s, p) for s in smiles if (p := fast_score(s)) is not None]
        if not props:
            return []

        if self.workers > 1 and len(props) > 1:
            self._ensure_pool()
            args = [(smi, self.receptor_pdb, self.center, self.size,
                     self.exhaustiveness, self.smina_cpu) for smi, _ in props]
            affs = self._pool.map(_dock_one, args)
        else:
            affs = [docking.dock(smi, self.receptor_pdb, center=self.center, size=self.size,
                                 exhaustiveness=self.exhaustiveness, cpu=self.smina_cpu)["affinity"]
                    for smi, _ in props]

        out = []
        for (smi, p), aff in zip(props, affs):
            p["affinity"] = aff
            p["aggregate"] = scoring.aggregate(p)
            out.append(p)
        return out

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
