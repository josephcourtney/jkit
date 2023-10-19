#%%
import os

import jkit.warn_with_traceback

jkit.warn_with_traceback.enable()

os.environ["MMTBX_CCP4_MONOMER_LIB"] = "/usr/local/MolProbity/modules/chem_data/mon_lib"
os.environ["CLIBD_MON"] = "/usr/local/MolProbity/modules/chem_data/geostd"


import libtbx.load_env

libtbx.env.add_repository(
    libtbx.path.relocatable_path(libtbx.path.absolute_path("/usr/local/MolProbity/modules/"), ".")
)

from mmtbx.validation.restraints import chiralities

from io import StringIO
from iotbx import pdb
from mmtbx.validation import cbetadev, omegalyze, ramalyze, rotalyze
from mmtbx.validation.molprobity import mp_geo
import subprocess
import pandas as pd
from jkit.pdb import PDB
import tempfile
import numpy as np
from scipy.special import erfinv, gammaln


def find_clashes(pdb_path, add_protons=False):
    # write pdb to temporary file

    prot = PDB.from_file(pdb_path)
    n_h = len(prot.models[0].df[prot.models[0].df.atom_name.str[0] == "H"])
    n_atoms = len(prot.models[0].df)
    if n_h / n_atoms < 0.4 and not add_protons:
        raise Warning(f"{pdb_path} is contains {100*n_h/n_atoms:6.2f}% protons")

    if add_protons:
        p = subprocess.run(
            [f"/usr/local/phenix-1.19.1-4122/build/bin/reduce -FLIP {pdb_path}",],
            capture_output=True,
            check=True,
            shell=True,
        )
        p_with_h = tempfile.NamedTemporaryFile()
        p_with_h.write(p.stdout)
        p_with_h.seek(0)
        pdb_path = p_with_h.name

        prot = PDB.from_file(pdb_path)
        n_h = len(prot.models[0].df[prot.models[0].df.atom_name.str[0] == "H"])
        n_atoms = len(prot.models[0].df)

    p = subprocess.run(
        [
            '/usr/local/phenix-1.19.1-4122/build/bin/probe -u -q -mc -het -once -NOVDWOUT -CON -nuclear "ogt10 not water" "ogt10" '
            + pdb_path,
        ],
        capture_output=True,
        check=True,
        shell=True,
    )

    if add_protons:
        p_with_h.close()

    s = p.stdout.decode("utf-8").split("\n")

    rows = []
    for line in s:
        cols = line.split(":")
        if len(cols) != 20:
            continue
        source, target = sorted([cols[3], cols[4]])
        rows.append(
            dict(
                probe_type=cols[2],
                chain_id_source=source[0:2],
                resseq_source=int(source[2:6]),
                icode_source=source[6],
                resname_source=source[7:10],
                altloc_source=source[15],
                name_source=source[11:15],
                chain_id_target=target[0:2],
                resseq_target=int(target[2:6]),
                icode_target=target[6],
                resname_target=target[7:10],
                altloc_target=target[15],
                name_target=target[11:15],
                overlap=float(cols[6]),
                x=float(cols[-5]),
                y=float(cols[-4]),
                z=float(cols[-3]),
            )
        )

    contacts = pd.DataFrame(rows)
    identifying_keys = [
        "chain_id_target",
        "resseq_target",
        "icode_target",
        "resname_target",
        "altloc_target",
        "name_target",
        "chain_id_source",
        "resseq_source",
        "icode_source",
        "resname_source",
        "altloc_source",
        "name_source",
    ]
    gb = contacts.groupby(["probe_type"] + identifying_keys, sort=False)
    idx = gb["overlap"].transform(min) == contacts["overlap"]
    contacts = contacts[idx]

    clashes = contacts[contacts.overlap <= -0.4]

    hbonds = clashes.copy()[clashes["probe_type"] == "hb"]
    hbonds.loc[:, "key"] = [tuple(r) for i, r in hbonds[identifying_keys].iterrows()]

    clashes = clashes.copy()[clashes["probe_type"] != "hb"]
    clashes.loc[:, "key"] = [tuple(r) for i, r in clashes[identifying_keys].iterrows()]

    common_df = pd.merge(clashes, hbonds, how="inner", on="key")
    clashes = clashes[~clashes["key"].isin(common_df["key"])]
    clashes = clashes.drop_duplicates("key").drop("key", axis=1).sort_values("overlap").reset_index(drop=True)
    clashscore = len(clashes.index) * 1e3 / n_atoms
    return clashes, clashscore, n_atoms


def molprobity_report(fname):
    pdb_io = pdb.input(fname)
    hierarchy = pdb_io.construct_hierarchy()
    buffer = StringIO()

    vdw_overlaps, clashscore, n_atoms = find_clashes(fname, add_protons=True)

    ramalyze_output = ramalyze.ramalyze(pdb_hierarchy=hierarchy, outliers_only=False, out=buffer)
    backbone_dihedral = pd.DataFrame(
        [
            dict(
                chain_id=r.chain_id,
                resseq=int(r.resseq),
                icode=r.icode,
                altloc=r.altloc,
                resname=r.resname,
                phi=r.phi,
                psi=r.psi,
                outlier=r.outlier,
                ramalyze_type=r.ramalyze_type(),
                score=r.score,
            )
            for r in ramalyze_output.results
        ]
    )
    backbone_dihedral_summary = dict(
        n_favored=ramalyze_output.n_favored,
        n_allowed=ramalyze_output.n_allowed,
        n_outliers=ramalyze_output.n_outliers,
        n_total=ramalyze_output.n_total,
    )

    omegalyze_output = omegalyze.omegalyze(pdb_hierarchy=hierarchy, nontrans_only=False, out=buffer, quiet=False,)
    backbone_omega = pd.DataFrame(
        [
            dict(
                chain_id=r.chain_id,
                resseq=r.resseq,
                icode=r.icode,
                altloc=r.altloc,
                resname=r.resname,
                prev_resseq=r.prev_resseq,
                prev_icode=r.prev_icode,
                prev_altloc=r.prev_altloc,
                prev_resname=r.prev_resname,
                omega=r.omega,
                omega_type=r.omega_type,
                is_nontrans=r.is_nontrans,
            )
            for r in omegalyze_output.results
        ]
    )
    backbone_omega_summary = dict(
        n_cis_proline=omegalyze_output.n_cis_proline(),
        n_twisted_proline=omegalyze_output.n_twisted_proline(),
        n_proline=omegalyze_output.n_proline(),
        n_cis_general=omegalyze_output.n_cis_general(),
        n_twisted_general=omegalyze_output.n_twisted_general(),
        n_general=omegalyze_output.n_general(),
    )

    cbetadev_output = cbetadev.cbetadev(pdb_hierarchy=hierarchy, outliers_only=False, out=buffer)
    cbeta_orientation = pd.DataFrame(
        [
            dict(
                chain_id=r.chain_id,
                resseq=r.resseq,
                icode=r.icode,
                altloc=r.altloc,
                resname=r.resname,
                deviation=r.deviation,
                dihedral_NABB=r.dihedral_NABB,
                outlier=r.outlier,
            )
            for r in cbetadev_output.results
        ]
    )
    cbeta_orientation_summary = dict(n_outliers=cbetadev_output.n_outliers)

    rotalyze_output = rotalyze.rotalyze(pdb_hierarchy=hierarchy, outliers_only=False, out=buffer)
    rotamer = pd.DataFrame(
        [
            dict(
                chain_id=r.chain_id,
                resseq=r.resseq,
                icode=r.icode,
                altloc=r.altloc,
                resname=r.resname,
                chi_angles=r.chi_angles,
                outlier=r.outlier,
                incomplete=r.incomplete,
                rotamer_name=r.rotamer_name,
                score=r.score,
            )
            for r in rotalyze_output.results
        ]
    )
    rotamer_summary = dict(n_outliers=rotalyze_output.n_outliers)

    with tempfile.NamedTemporaryFile() as f:
        mp_geo.run(
            [f"pdb={fname}", f"out_file={f.name}", "outliers_only=False", "bonds_and_angles=True",]
        )
        f.seek(0)
        mp_geo_output = f.read().decode("utf-8").split("\n")

    chiralities = []
    bonds = []
    angles = []
    for line in mp_geo_output:
        cols = line.split(":")
        if len(cols) != 10:
            continue

        r = dict(
            chain_id=cols[1].strip(),
            resseq=int(cols[2]),
            icode=cols[3],
            altloc=cols[4],
            resname=cols[5],
            score=float(cols[8]),
        )

        atoms_str = cols[6]
        atoms_involved = [a for a in atoms_str.split("-") if len(a) > 0]
        if len(atoms_involved) == 1:
            r["atom_0"] = atoms_involved[0]
            r["chiral_volume"] = float(cols[7])
            chiralities.append(r)
        elif len(atoms_involved) == 2:
            r["atom_0"] = atoms_involved[0]
            r["atom_1"] = atoms_involved[1]
            r["bond_length"] = float(cols[7])
            bonds.append(r)
        elif len(atoms_involved) == 3:
            r["atom_0"] = atoms_involved[0]
            r["atom_1"] = atoms_involved[1]
            r["atom_2"] = atoms_involved[2]
            r["bond_angle"] = float(cols[7])
            angles.append(r)

    chiral_volume = pd.DataFrame(chiralities)
    bond_length = pd.DataFrame(bonds)
    bond_angle = pd.DataFrame(angles)

    def icdf(p):
        return np.sqrt(2) * erfinv(2 * p - 1)

    cutoff_z_score = 3

    z_vdw = np.sum(cutoff_z_score * len(vdw_overlaps.index))
    n_vdw = 5 * n_atoms

    z_bond_length = np.sum(np.abs(np.clip(bond_length.score, a_min=-6, a_max=6)))
    n_bond_length = len(bond_length.index)

    z_bond_angle = np.sum(np.abs(np.clip(bond_angle.score, a_min=-6, a_max=6)))
    n_bond_angle = len(bond_angle.index)

    z_backbone_dihedral = np.sum(np.abs(np.clip(icdf(backbone_dihedral.score / 100), a_min=-6, a_max=6)))
    n_backbone_dihedral = len(backbone_dihedral.index)

    z_backbone_omega = cutoff_z_score * backbone_omega.is_nontrans.sum()
    n_backbone_omega = len(backbone_omega.index)

    z_cbeta_orientation = cutoff_z_score * cbeta_orientation.outlier.sum()
    n_cbeta_orientation = len(cbeta_orientation.index)

    z_rotamer = np.sum(np.abs(np.clip(icdf(rotamer.score / 100), a_min=-6, a_max=6)))
    n_rotamer = len(rotamer.index)

    z_chiral_volume = np.sum(np.abs(np.clip(chiral_volume.score, a_min=-6, a_max=6)))
    n_chiral_volume = len(chiral_volume.index)

    n = (
        n_vdw
        + n_bond_length
        + n_bond_angle
        + n_backbone_dihedral
        + n_backbone_omega
        + n_cbeta_orientation
        + n_rotamer
        + n_chiral_volume
    )

    # combine z-scores with Stouffer's method
    z_composite = (
        z_vdw
        + z_bond_length
        + z_bond_angle
        + z_backbone_dihedral
        + z_backbone_omega
        + z_cbeta_orientation
        + z_rotamer
        + z_chiral_volume
    ) / n

    return (
        backbone_dihedral_summary
        | backbone_omega_summary
        | cbeta_orientation_summary
        | rotamer_summary
        | {
            "clashscore": clashscore,
            "fname": fname,
            "z_vdw": (z_vdw / n_vdw),
            "z_bond_length": (z_bond_length / n_bond_length),
            "z_bond_angle": (z_bond_angle / n_bond_angle),
            "z_backbone_dihedral": (z_backbone_dihedral / n_backbone_dihedral),
            "z_backbone_omega": (z_backbone_omega / n_backbone_omega),
            "z_cbeta_orientation": (z_cbeta_orientation / n_cbeta_orientation),
            "z_rotamer": (z_rotamer / n_rotamer),
            "z_chiral_volume": (z_chiral_volume / n_chiral_volume),
            "z_composite": z_composite,
        }
    )


if __name__ == '__main__':
    from rich import print

    fname = "./1yk4.pdb"
    print(molprobity_report(fname))

#%%

"""
add protons
calculate
    nearest neighbor distances (Delaunay connectivity)
        cluster (residue_0, atom_0, residue_1, atom_1)
    bond lengths
        cluster (residue, atom_0, atom_1)
        identify atom equivalence by overlay of element and hybridization
    bond angles
        cluster (residue, atom_0, atom_1, atom_2)
        identify atom equivalence by overlay of element and hybridization
    phi, psi, omega
        discretize omega into trans / cis
        splits
            residue specific
            general, glycine, proline, pre-proline
            residue triplet
    cbeta location as 3D distribution over
        average of (N-CA-CB) and (C-CA-CB) angles
        average of (C-N-CA-CB) and (N-C-CA-CB) dihedrals
        (CA-CB) bond length
    rotamers as residue-specific chi dihedral distribution
    chiral volumes
        (residue, atom)-specific

cluster distributions with pairwise mean of absolute differences between emprical CDFs
use convex hull (alpha complex) peeling to define regions containing successively more of the distribution
use layers to define the empirical PDF of the distribution as a function of #layers from the center
    interpolate position in between layers to get a fractional layer coordinate
get p-values per-atom, distance, angle, dihedral, chirality
get composite p-values for each data type
compute weighted harmonic mean of composite p values




"""
