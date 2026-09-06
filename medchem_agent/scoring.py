"""Multi-objective scoring combining docking, drug-likeness, synthesizability, ADMET.

All objectives are normalized to [0, 1] (higher = better) before aggregation so
that a single scalar can drive greedy/beam iteration. The Pareto front is
tracked separately so the paper can report multi-objective trade-offs.
"""
from . import config
from .tools import rdkit_tools

# normalization bounds (physically sensible ranges)
DOCK_BEST = -11.0   # kcal/mol, very strong binder -> maps to 1.0
DOCK_WORST = -4.0   # kcal/mol, weak -> maps to 0.0


def _norm_docking(affinity):
    """Lower kcal/mol is better; map [DOCK_WORST, DOCK_BEST] -> [0, 1]."""
    if affinity is None:
        return 0.0
    v = (affinity - DOCK_WORST) / (DOCK_BEST - DOCK_WORST)
    return max(0.0, min(1.0, v))


def _norm_sa(sa):
    """SA in [1,10]; 1 (easy) -> 1.0, 10 (hard) -> 0.0."""
    if sa is None:
        return 0.0
    return max(0.0, min(1.0, (10.0 - sa) / 9.0))


def _norm_admet(admet_pass, admet_total):
    if not admet_total:
        return 0.0
    return admet_pass / admet_total


def aggregate(properties: dict, weights: dict = None) -> float:
    """properties must contain: qed, sa, affinity, admet_pass, admet_total."""
    weights = weights or {
        "docking": config.get("scoring", "docking_weight", default=0.35),
        "qed": config.get("scoring", "qed_weight", default=0.25),
        "sa": config.get("scoring", "sa_weight", default=0.20),
        "admet": config.get("scoring", "admet_weight", default=0.20),
    }
    s_dock = _norm_docking(properties.get("affinity"))
    s_qed = properties.get("qed", 0.0) or 0.0
    s_sa = _norm_sa(properties.get("sa"))
    s_admet = _norm_admet(properties.get("admet_pass", 0), properties.get("admet_total", 0))
    return (
        weights["docking"] * s_dock
        + weights["qed"] * s_qed
        + weights["sa"] * s_sa
        + weights["admet"] * s_admet
    )


def is_dominated(a, b):
    """True if b dominates a (b is >= a on every objective and > on at least one).
    Objectives (higher=better): qed, norm_sa, norm_docking, norm_admet."""
    a_vec = (a["qed"], _norm_sa(a["sa"]), _norm_docking(a["affinity"]),
             _norm_admet(a.get("admet_pass", 0), a.get("admet_total", 0)))
    b_vec = (b["qed"], _norm_sa(b["sa"]), _norm_docking(b["affinity"]),
             _norm_admet(b.get("admet_pass", 0), b.get("admet_total", 0)))
    return all(bv >= av for av, bv in zip(a_vec, b_vec)) and any(bv > av for av, bv in zip(a_vec, b_vec))


def pareto_front(candidates: list):
    """Return the non-dominated subset of candidates (list of property dicts)."""
    front = []
    for c in candidates:
        if not any(is_dominated(c, f) for f in front):
            front = [f for f in front if not is_dominated(f, c)]
            front.append(c)
    return front
