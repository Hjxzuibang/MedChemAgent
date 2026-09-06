"""Pareto-front helpers operating on RAW objectives.

The scalar ``aggregate`` in :mod:`medchem_agent.scoring` maps docking through
``_norm_docking``, which clamps everything <= DOCK_BEST (-11 kcal/mol) to 1.0.
On deep-binding targets (CDK2 -12.5, ABL1 -12.8) this saturates the docking
term, so the scalar cannot distinguish the starting hit from a genuine -14
kcal/mol improvement -- the agent is effectively blind to affinity gains beyond
-11. These helpers keep the RAW docking score and a renormalized drug-likeness
blend, so the Pareto front (and its hypervolume) reflects real affinity headroom.
"""
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from . import config, scoring

DOCK_REF = -4.0   # worst docking considered "acceptable" (matches DOCK_WORST)
DL_REF = 0.0      # worst drug-likeness

_FP_GEN = None


def _fp(mol):
    global _FP_GEN
    if _FP_GEN is None:
        _FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _FP_GEN.GetFingerprint(mol)


def druglikeness(c: dict) -> float:
    """Non-docking blend of QED/SA/ADMET, renormalized to [0,1] (higher=better)."""
    w_qed = config.get("scoring", "qed_weight", default=0.25)
    w_sa = config.get("scoring", "sa_weight", default=0.20)
    w_admet = config.get("scoring", "admet_weight", default=0.20)
    tot = w_qed + w_sa + w_admet
    s_qed = c.get("qed", 0.0) or 0.0
    s_sa = scoring._norm_sa(c.get("sa"))
    s_admet = scoring._norm_admet(c.get("admet_pass", 0), c.get("admet_total", 0))
    return (w_qed * s_qed + w_sa * s_sa + w_admet * s_admet) / tot


def pareto_front_raw(candidates: list) -> list:
    """Non-dominated subset over (docking=minimize, druglikeness=maximize)."""
    scored = [(c, c["affinity"], druglikeness(c))
              for c in candidates if c.get("affinity") is not None]
    front = []
    for c, dock, dl in scored:
        # c is dominated if some front member beats it on both (one strict)
        if any(fd <= dock and fdl >= dl and (fd < dock or fdl > dl)
               for _, fd, fdl in front):
            continue
        front = [(fc, fd, fdl) for fc, fd, fdl in front
                 if not (dock <= fd and dl >= fdl and (dock < fd or dl > fdl))]
        front.append((c, dock, dl))
    return [fc for fc, _, _ in front]


def hypervolume(front: list, dock_ref: float = DOCK_REF, dl_ref: float = DL_REF) -> float:
    """2D hypervolume (area) dominated by ``front`` above reference (dock_ref, dl_ref).

    Transforms docking -> -docking so both axes maximize, then integrates the
    step function of coverage. Reference = (dock_ref, dl_ref), the worst
    acceptable docking / drug-likeness; points not improving on it are ignored.
    """
    pts = sorted((-c["affinity"], druglikeness(c))
                 for c in front if c.get("affinity") is not None)
    ref_x, ref_y = -dock_ref, dl_ref
    pts = [(x, y) for x, y in pts if x >= ref_x and y >= ref_y]
    if not pts:
        return 0.0
    n = len(pts)
    # suffix max over y (coverage_y(x) for x just below each point)
    suffix = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = max(pts[i][1], suffix[i + 1])
    area = (pts[0][0] - ref_x) * (suffix[0] - ref_y)
    for i in range(1, n):
        area += (pts[i][0] - pts[i - 1][0]) * (suffix[i] - ref_y)
    return area


def tanimoto(smi_a: str, smi_b: str) -> float:
    ma = Chem.MolFromSmiles(smi_a)
    mb = Chem.MolFromSmiles(smi_b)
    if ma is None or mb is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(_fp(ma), _fp(mb))


def novelty(smi: str, archive: list) -> float:
    """1 - max Morgan-Tanimoto to the archive; 1.0 if archive empty."""
    if not archive:
        return 1.0
    return 1.0 - max(tanimoto(smi, t) for t in archive)


def diversity(candidates: list) -> float:
    """Mean pairwise (1 - Tanimoto); higher = more chemically diverse."""
    smis = [c.get("smiles") for c in candidates if c.get("smiles")]
    if len(smis) < 2:
        return 1.0
    tot, n = 0.0, 0
    for i in range(len(smis)):
        for j in range(i + 1, len(smis)):
            tot += 1.0 - tanimoto(smis[i], smis[j])
            n += 1
    return tot / n


def docking_best_gated(candidates: list, qed_min: float = 0.5, sa_max: float = 4.0):
    """Docking-primary selection among candidates passing drug-likeness gates."""
    gated = [c for c in candidates if c.get("affinity") is not None
             and (c.get("qed") or 0.0) >= qed_min and (c.get("sa") or 99.0) <= sa_max]
    return min(gated, key=lambda c: c["affinity"]) if gated else None
