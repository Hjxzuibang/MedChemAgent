"""ADMET / medicinal-chemistry filters (rule-based, self-contained).

Uses the standard filter panel applied in hit-to-lead:
  - Lipinski rule-of-five (oral drug-likeness)
  - Veber rule (oral bioavailability: rotatable bonds <= 10, TPSA <= 140)
  - PAINS / Brenk / NIH substructure alerts via RDKit FilterCatalog
    (Baell & Holloway 2010; Brenk et al. 2008)

Returns (pass_count, total_count), normalized to [0,1] by the scorer.
Self-contained: no ML dependencies; rule filters are what medicinal chemists use.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams

_CATALOG = None


def _get_catalog():
    global _CATALOG
    if _CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
        _CATALOG = FilterCatalog.FilterCatalog(params)
    return _CATALOG


def _lipinski_ok(mol) -> bool:
    return (
        Descriptors.MolWt(mol) <= 500
        and Descriptors.MolLogP(mol) <= 5
        and rdMolDescriptors.CalcNumHBD(mol) <= 5
        and rdMolDescriptors.CalcNumHBA(mol) <= 10
    )


def _veber_ok(mol) -> bool:
    return (
        rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
        and Descriptors.TPSA(mol) <= 140
    )


def score_smiles(smiles: str):
    """Return (pass_count, total_count) over {Lipinski, Veber, no-alerts}."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0, 0

    total = 3
    passed = 0
    if _lipinski_ok(m):
        passed += 1
    if _veber_ok(m):
        passed += 1
    if len(_get_catalog().GetMatches(m)) == 0:
        passed += 1
    return passed, total
