"""RDKit-based molecular descriptors, drug-likeness and synthesizability.

Provides a single `score_molecule(smi)` entry point returning a dict of
properties used by the multi-objective scorer:
  - validity / canonical SMILES
  - QED (drug-likeness, 0..1)
  - SA score (Ertl & Schuffenhauer 2009; 1=easy, 10=hard)
  - Lipinski rule-of-five violations
  - basic descriptors (MW, logP, TPSA, HBD/HBA, rotatable bonds, rings)
"""
import gzip
import math
import os
import pickle

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, RDConfig, rdFingerprintGenerator, rdMolDescriptors

# --------------------------------------------------------------------------- #
# SA score (vendored from Ertl & Schuffenhauer, J. Cheminf. 2009, RDKit contrib)
# --------------------------------------------------------------------------- #
_FSCORES = None


def _read_fragment_scores():
    global _FSCORES
    if _FSCORES is not None:
        return _FSCORES
    path = os.path.join(RDConfig.RDContribDir, "SA_Score", "fpscores.pkl.gz")
    with gzip.open(path, "rb") as f:
        data = pickle.load(f)
    # fpscores.pkl is a list of rows [score, bit_id, bit_id, ...]
    _FSCORES = {}
    for row in data:
        for bit_id in row[1:]:
            _FSCORES[bit_id] = float(row[0])
    return _FSCORES


def sa_score(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None

    # fragment score
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
    sfp = fp_gen.GetSparseCountFingerprint(m)
    nze = sfp.GetNonzeroElements()
    fscores = _read_fragment_scores()
    score1 = 0.0
    nf = 0
    for bit_id, count in nze.items():
        nf += count
        score1 += fscores.get(bit_id, -4.0) * count
    score1 /= nf

    # feature penalty
    n_atoms = m.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(m)
    n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(m)
    n_macro = sum(1 for ring in ri.AtomRings() if len(ring) > 8)

    size_pen = n_atoms ** 1.005 - n_atoms
    stereo_pen = math.log10(n_chiral + 1)
    spiro_pen = math.log10(n_spiro + 1)
    bridge_pen = math.log10(n_bridge + 1)
    macro_pen = math.log10(2) if n_macro > 0 else 0.0
    score2 = 0.0 - size_pen - stereo_pen - spiro_pen - bridge_pen - macro_pen

    # fingerprint-density correction (added in sascorer 1.1)
    num_bits = len(nze)
    if n_atoms > num_bits:
        score3 = math.log(float(n_atoms) / num_bits) * 0.5
    else:
        score3 = 0.0

    sascore = score1 + score2 + score3

    # rescale raw value into [1, 10]
    lo, hi = -4.0, 2.5
    sascore = 11.0 - (sascore - lo + 1) / (hi - lo) * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0
    return sascore


# --------------------------------------------------------------------------- #
# Lipinski rule of five
# --------------------------------------------------------------------------- #
def lipinski_violations(mol) -> int:
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    n = 0
    if mw > 500:
        n += 1
    if logp > 5:
        n += 1
    if hbd > 5:
        n += 1
    if hba > 10:
        n += 1
    return n


def canonical_smiles(smi: str):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else None


def is_valid(smi: str) -> bool:
    return Chem.MolFromSmiles(smi) is not None


# --------------------------------------------------------------------------- #
# Combined property computation
# --------------------------------------------------------------------------- #
def score_molecule(smi: str):
    """Return a dict of properties, or None if the SMILES is invalid."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    can = Chem.MolToSmiles(m, isomericSmiles=True)
    return {
        "smiles": can,
        "valid": True,
        "mw": Descriptors.MolWt(m),
        "logp": Crippen.MolLogP(m),
        "tpsa": Descriptors.TPSA(m),
        "hbd": rdMolDescriptors.CalcNumHBD(m),
        "hba": rdMolDescriptors.CalcNumHBA(m),
        "rotatable": rdMolDescriptors.CalcNumRotatableBonds(m),
        "rings": rdMolDescriptors.CalcNumRings(m),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
        "qed": QED.qed(m),
        "sa": sa_score(can),
        "lipinski": lipinski_violations(m),
    }
