"""Docking via smina (AutoDock Vina fork).

smina reads a PDB receptor and an SDF ligand directly, adding hydrogens and
atom types itself, so no OpenBabel/Meeko prep is required. Returns a Vina
affinity (kcal/mol, lower = better binding).
"""
import os
import shutil
import subprocess
import tempfile

from rdkit import Chem
from rdkit.Chem import AllChem

from .. import config


def _smina_bin() -> str:
    return os.environ.get("SMINA_BIN") or config.get("paths", "smina_bin", default="smina")


def smiles_to_sdf(smi: str, path: str, seed: int = 42) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"invalid SMILES: {smi}")
    m = Chem.AddHs(m)
    # EmbedMolecule returns 0 on success, -1 on failure. Some highly strained or
    # unkekulizable structures cannot be embedded; fall back to random coords
    # before giving up so a bad proposal degrades to "no score" instead of crash.
    ok = AllChem.EmbedMolecule(m, randomSeed=seed)
    if ok != 0:
        ok = AllChem.EmbedMolecule(m, randomSeed=seed, useRandomCoords=True)
    if ok != 0:
        raise ValueError(f"3D embedding failed for: {smi}")
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        pass  # MMFF may fail on exotic topologies; the raw conformer is still usable
    w = Chem.SDWriter(path)
    w.write(m)
    w.close()
    return path


def _read_best_score(sdf_path: str):
    """Return the best (lowest) affinity from a multi-pose smina output SDF."""
    best = None
    try:
        with Chem.rdBase.BlockLogs():  # silence "2D but has Z" warnings from smina SDF
            supp = Chem.SDMolSupplier(sdf_path)
            for m in supp:
                if m is None:
                    continue
                for prop in ("minimizedAffinity", "CNNaffinity", "CNNscore"):
                    if m.HasProp(prop):
                        try:
                            val = float(m.GetProp(prop))
                        except ValueError:
                            continue
                        if best is None or val < best:
                            best = val
                        break
    except (OSError, ValueError):
        # smina may write a truncated/corrupt SDF (e.g. a disk-quota hit mid-write);
        # treat it as "no score" rather than crashing the whole campaign.
        return None
    return best


def dock(smi: str, receptor_pdb: str, ref_ligand_sdf: str = None,
         center: tuple = None, size: tuple = (20.0, 20.0, 20.0),
         exhaustiveness: int = None, cpu: int = None, seed: int = 42,
         workdir: str = None, cleanup: bool = True):
    """Dock `smi` into `receptor_pdb`.

    Box is defined either by an explicit `center`/`size` (Angstrom) or by the
    `ref_ligand_sdf` (autobox). Returns a dict with keys `affinity` and `workdir`.
    """
    exhaustiveness = exhaustiveness or config.get("agent", "exhaustiveness", default=8)
    cpu = cpu or config.get("agent", "smina_cpu", default=4)
    workdir = workdir or tempfile.mkdtemp(prefix="dock_")
    try:
        lig_sdf = smiles_to_sdf(smi, os.path.join(workdir, "lig.sdf"), seed=seed)
    except Exception as e:  # noqa: BLE001 - un-embeddable proposal -> no docking score
        if cleanup:
            shutil.rmtree(workdir, ignore_errors=True)
        return {"affinity": None, "workdir": None, "stderr": f"embed failed: {e}"}
    out_sdf = os.path.join(workdir, "docked.sdf")

    cmd = [
        _smina_bin(), "--receptor", receptor_pdb, "--ligand", lig_sdf,
        "--out", out_sdf, "--exhaustiveness", str(exhaustiveness),
        "--cpu", str(cpu), "--seed", str(seed),
    ]
    if center is not None:
        cmd += ["--center_x", f"{center[0]:.3f}",
                "--center_y", f"{center[1]:.3f}",
                "--center_z", f"{center[2]:.3f}",
                "--size_x", f"{size[0]:.1f}",
                "--size_y", f"{size[1]:.1f}",
                "--size_z", f"{size[2]:.1f}"]
    elif ref_ligand_sdf is not None:
        cmd += ["--autobox_ligand", ref_ligand_sdf]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if (proc.returncode != 0 or not os.path.exists(out_sdf)
            or os.path.getsize(out_sdf) == 0):
        # smina exited 0 but produced no poses (or wrote an empty file, e.g. on a
        # transient disk/quota failure); report "no score" instead of crashing.
        return {"affinity": None, "workdir": workdir, "stderr": proc.stderr.strip()[-500:]}

    affinity = _read_best_score(out_sdf)
    if cleanup:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"affinity": affinity, "workdir": None if cleanup else workdir, "stderr": None}
