#!/usr/bin/env python
#%%
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import PDBList
import pdbfixer
from simtk.openmm import app
import jkit.aminoacids
import numpy as np
import pandas as pd


class StateSet:
    # type
    # trajectory
    # ensemble
    # markov state model
    # shared information
    # experimental details
    # unit cell
    # periodic box

    # denote whether this is an asymmetric unit or the full universe

    def __init__(self, name="", pdb_id="", unit_cell=None, resolution=None):
        self.name = name
        self.pdb_id = pdb_id

        self.unit_cell = unit_cell
        self.resolution = resolution

        self._assemblies = []
        self._asm_index = {}

    def add_State(self, asm):
        self._asm_index[asm.index] = len(self._assemblies)
        self._assemblies.append(asm)

    def __getitem__(self, index):
        return self._assemblies[self._asm_index[index]]

    @classmethod
    def from_pdbx_mmcif(cls, f, name=""):
        _dict = MMCIF2Dict(f)

        def _mmcif_get(keys):
            try:
                return {
                    e
                    for k in keys
                    for e in _dict.get(k, [])
                    if e is not None and e != "?"
                }.pop()
            except KeyError:
                return None

        pdb_id = _mmcif_get(["_entry.id"])

        unit_cell = {
            "a": np.float64(_mmcif_get(["_cell.length_a"])),
            "a_err": np.float64(_mmcif_get(["_cell.length_a_esd"])),
            "b": np.float64(_mmcif_get(["_cell.length_b"])),
            "b_err": np.float64(_mmcif_get(["_cell.length_b_esd"])),
            "c": np.float64(_mmcif_get(["_cell.length_c"])),
            "c_err": np.float64(_mmcif_get(["_cell.length_c_esd"])),
            "alpha": np.float64(_mmcif_get(["_cell.angle_alpha"])),
            "alpha_err": np.float64(_mmcif_get(["_cell.angle_alpha_esd"])),
            "beta": np.float64(_mmcif_get(["_cell.angle_beta"])),
            "beta_err": np.float64(_mmcif_get(["_cell.angle_beta_esd"])),
            "gamma": np.float64(_mmcif_get(["_cell.angle_gamma"])),
            "gamma_err": np.float64(_mmcif_get(["_cell.angle_gamma_esd"])),
            "n_per_cell": int(_mmcif_get(["_cell.Z_PDB"])),
            "unique_axis": str(_mmcif_get(["_cell.pdbx_unique_axis"])),
        }

        name = str(_mmcif_get(["_struct.title"]))

        resolution = np.float64(
            _mmcif_get(
                [
                    "_refine.ls_d_res_high",
                    "_refine_hist.d_res_high",
                    "_em_3d_reconstruction.resolution",
                ]
            )
        )

        df_atom = pd.DataFrame(
            {
                "record_type": _dict.get("_atom_site.group_PDB", None),
                "id": _dict.get("_atom_site.id", None),
                "element": _dict.get("_atom_site.type_symbol", None),
                "atom_type": _dict.get("_atom_site.label_atom_id", None),
                "alt_id": _dict.get("_atom_site.label_alt_id", None),
                "residue_type": _dict.get("_atom_site.label_comp_id", None),
                "asym_id": _dict.get("_atom_site.label_asym_id", None),
                "entity_id": _dict.get("_atom_site.label_entity_id", None),
                "residue_number": _dict.get("_atom_site.label_seq_id", None),
                "insertion_code": _dict.get("_atom_site.pdbx_PDB_ins_code", None),
                "x": _dict.get("_atom_site.Cartn_x", None),
                "y": _dict.get("_atom_site.Cartn_y", None),
                "z": _dict.get("_atom_site.Cartn_z", None),
                "occupancy": _dict.get("_atom_site.occupancy", None),
                "b": _dict.get("_atom_site.B_iso_or_equiv", None),
                "formal_charge": _dict.get("_atom_site.pdbx_formal_charge", None),
                "model_num": _dict.get("_atom_site.pdbx_PDB_model_num", None),
            }
        ).replace("?", np.nan)
        df_atom["id"] = pd.to_numeric(df_atom["id"], errors="coerce")
        df_atom["entity_id"] = pd.to_numeric(df_atom["entity_id"], errors="coerce")
        df_atom["residue_number"] = pd.to_numeric(
            df_atom["residue_number"], errors="coerce"
        )
        df_atom["formal_charge"] = pd.to_numeric(
            df_atom["formal_charge"], errors="coerce"
        )
        df_atom["model_num"] = pd.to_numeric(df_atom["model_num"], errors="coerce")
        df_atom["x"] = pd.to_numeric(df_atom["x"], errors="coerce")
        df_atom["y"] = pd.to_numeric(df_atom["y"], errors="coerce")
        df_atom["z"] = pd.to_numeric(df_atom["z"], errors="coerce")
        df_atom["occupancy"] = pd.to_numeric(df_atom["occupancy"], errors="coerce")
        df_atom["b"] = pd.to_numeric(df_atom["b"], errors="coerce")

        df_mol = pd.merge(
            pd.DataFrame(
                {
                    "entity_id": _dict.get("_entity.id", None),
                    "mol_type": _dict.get("_entity.type", None),
                    "description": _dict.get("_entity.pdbx_description", None),
                    "formula_weight": _dict.get("_entity.formula_weight", None),
                    "n_molecules": _dict.get("_entity.pdbx_number_of_molecules", None),
                }
            ),
            pd.DataFrame(
                {
                    "entity_id": _dict.get("_entity_poly.entity_id", None),
                    "poly_type": _dict.get("_entity_poly.type", None),
                    "nstd_linkage": _dict.get("_entity_poly.nstd_linkage", None),
                    "nstd_monomer": _dict.get("_entity_poly.nstd_monomer", None),
                    "sequence": _dict.get("_entity_poly.pdbx_seq_one_letter_code_can", None),
                    "strand_id": _dict.get("_entity_poly.pdbx_strand_id", None),
                }
            ),
            how="left",
            on="entity_id",
        )
        non_poly = {
            "asym_id": _dict.get("_pdbx_nonpoly_scheme.asym_id", None),
            "entity_id": _dict.get("_pdbx_nonpoly_scheme.entity_id", None),
            "residue_type": _dict.get("_pdbx_nonpoly_scheme.mon_id", None),
            "pdb_seq_num": _dict.get("_pdbx_nonpoly_scheme.pdb_seq_num", None),
            "strand_id": _dict.get("_pdbx_nonpoly_scheme.pdb_strand_id", None),
            "insertion_code": _dict.get("_pdbx_nonpoly_scheme.pdb_ins_code", None),
        }
        non_poly = {k:v for k,v in non_poly.items() if v is not None}

        poly = {
            "asym_id": _dict.get("_pdbx_poly_seq_scheme.asym_id", None),
            "entity_id": _dict.get("_pdbx_poly_seq_scheme.entity_id", None),
            "residue_number": _dict.get("_pdbx_poly_seq_scheme.seq_id", None),
            "residue_type": _dict.get("_pdbx_poly_seq_scheme.mon_id", None),
            "pdb_seq_num": _dict.get("_pdbx_poly_seq_scheme.pdb_seq_num", None),
            "strand_id": _dict.get("_pdbx_poly_seq_scheme.pdb_strand_id", None),
            "insertion_code": _dict.get("_pdbx_poly_seq_scheme.pdb_ins_code", None),
            "heterogenous": _dict.get("_pdbx_poly_seq_scheme.hetero", None),
        }
        poly = {k:v for k,v in poly.items() if v is not None}

        dfs = [pd.DataFrame(e) for e in [non_poly, poly] if len(e.keys()) > 0]
        if len(dfs) > 0:
            df_res = (
                pd.concat(dfs)
                .replace("?", np.nan)
                .replace(".", np.nan)
            )
            df_res["entity_id"] = pd.to_numeric(df_res["entity_id"], errors="coerce")
            df_res["residue_number"] = pd.to_numeric(
                df_res["residue_number"], errors="coerce"
            )
            df_res["pdb_seq_num"] = pd.to_numeric(df_res["pdb_seq_num"], errors="coerce")
            # df_res["heterogenous"] = df_res["heterogenous"].astype("bool")
        else:
            df_res = pd.DataFrame(columns=["asym_id", "entity_id", "residue_number", "residue_type", "pdb_seq_num", "strand_id", "insertion_code", "heterogenous"])

        df_chem = pd.DataFrame(
            {
                "chem_id": _dict.get("_chem_comp.id", None),
                "chem_type": _dict["_chem_comp.type"],
                "nonstandard_residue": _dict["_chem_comp.mon_nstd_flag"],
                "chem_name": _dict["_chem_comp.name"],
                "chem_comp.formula": _dict["_chem_comp.formula"],
                "chem_comp.formula_weight": _dict["_chem_comp.formula_weight"],
            }
        )

        df_mol["n_molecules"] = pd.to_numeric(df_mol["n_molecules"], errors="coerce")
        df_mol["entity_id"] = pd.to_numeric(df_mol["entity_id"], errors="coerce")
        df_mol["formula_weight"] = pd.to_numeric(
            df_mol["formula_weight"], errors="coerce"
        )

        df_conn = pd.DataFrame(
            {
                "connection_type": _dict.get("_struct_conn.conn_type_id", None),
                "asym_id_0": _dict.get("_struct_conn.ptnr1_label_asym_id", None),
                "asym_id_1": _dict.get("_struct_conn.ptnr2_label_asym_id", None),
                "asym_id_2": _dict.get("_struct_conn.pdbx_ptnr3_label_asym_id", None),
                "residue_type_0": _dict.get("_struct_conn.ptnr1_label_comp_id", None),
                "residue_type_1": _dict.get("_struct_conn.ptnr2_label_comp_id", None),
                "residue_type_2": _dict.get("_struct_conn.pdbx_ptnr3_label_comp_id", None),
                "residue_number_0": _dict.get("_struct_conn.ptnr1_label_seq_id", None),
                "residue_number_1": _dict.get("_struct_conn.ptnr2_label_seq_id", None),
                "residue_number_2": _dict.get("_struct_conn.pdbx_ptnr3_label_seq_id", None),
                "ptnr1_label_atom_id": _dict.get("_struct_conn.ptnr1_label_atom_id", None),
                "ptnr2_label_atom_id": _dict.get("_struct_conn.ptnr2_label_atom_id", None),
                "pdbx_ptnr3_label_atom_id": _dict.get("_struct_conn.pdbx_ptnr3_label_atom_id", None),
                "alt_id_0": _dict.get("_struct_conn.pdbx_ptnr1_label_alt_id", None),
                "alt_id_1": _dict.get("_struct_conn.pdbx_ptnr2_label_alt_id", None),
                "alt_id_2": _dict.get("_struct_conn.pdbx_ptnr3_label_alt_id", None),
                "insertion_code_0": _dict.get("_struct_conn.pdbx_ptnr1_PDB_ins_code", None),
                "insertion_code_1": _dict.get("_struct_conn.pdbx_ptnr2_PDB_ins_code", None),
                "insertion_code_2": _dict.get("_struct_conn.pdbx_ptnr3_PDB_ins_code", None),
            }
        )

        s = cls(name=name, pdb_id=pdb_id, unit_cell=unit_cell, resolution=resolution)
        for n in df_atom["model_num"].unique():
            s.add_State(
                State.from_dataframes(
                    index=n,
                    df_atom=df_atom[df_atom["model_num"] == n],
                    df_mol=df_mol,
                    df_res=df_res,
                    df_chem=df_chem,
                    df_conn=df_conn,
                )
            )

        return s

    @classmethod
    def from_rcsb(cls, pdb_id, format="mmcif"):
        pdbl = PDBList(verbose=False)
        filename = pdbl.retrieve_pdb_file(pdb_id, file_format="mmCif")
        return cls.from_pdbx_mmcif(filename, name=pdb_id)

    @classmethod
    def from_sequence(cls, seq):
        seq = [jkit.aminoacids.one_to_three[e] for e in list(seq)]

        fixer = pdbfixer.PDBFixer(pdbid="1ubq")
        fixer.removeChains(list(range(fixer.topology.getNumChains()))[1:])

        chain = next(fixer.topology.chains())
        orig_len = len(list(chain.residues()))
        index = list(chain.residues())[-1].index

        fixer.missingResidues = {(chain.index, index + 1): seq}

        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        modeller = app.Modeller(fixer.topology, fixer.positions)
        chain = next(modeller.topology.chains())
        to_delete = []
        for res in list(chain.residues())[:orig_len]:
            for atom in res.atoms():
                to_delete.append(atom)
        modeller.delete(to_delete)

        with tempfile.TemporaryFile() as f:
            app.PDBxFile.writeFile(modeller.topology, modeller.positions, f)
            f.seek(0)
            return cls.from_pdbx_mmcif(f)

    def to_file(cls, path):
        raise NotImplementedError


class State:
    # molecules
    def __init__(self, index):
        self.index = index
        self._molecules = []
        self._mol_index = {}

    def add_molecule(self, mol):
        self._mol_index[mol.entity_id] = len(self._molecules)
        self._molecules.append(mol)

    def __getitem__(self, entity_id):
        return self._molecules[self._mol_index[entity_id]]

    @classmethod
    def from_dataframes(cls, index, df_atom, df_mol, df_res, df_chem, df_conn):
        a = cls(index=index)
        for i, r in df_mol.drop(columns=["n_molecules"]).iterrows():
            a.add_molecule(
                Molecule(
                    **(r.to_dict()),
                    atoms=df_atom[df_atom["entity_id"] == r.entity_id],
                    residues=df_res[df_res["entity_id"] == r.entity_id],
                    residue_definitions=df_chem,
                    bonds=df_conn,
                )
            )
        return a

    def fix(self):  # wrapper for pdbfixer
        raise NotImplementedError

    def mutate(self):  # wrapper for pdbfixer
        raise NotImplementedError

    def mutate(self):  # wrapper for pdbfixer
        raise NotImplementedError


class Molecule:
    def __init__(
        self,
        atoms,
        residues=None,
        residue_definitions=None,
        bonds=None,
        entity_id=-1,
        mol_type=None,
        description="",
        formula_weight=0.0,
        poly_type=None,
        nstd_linkage=None,
        nstd_monomer=None,
        sequence="",
        strand_id="",
    ):
        self.atoms = atoms
        self.residues = residues
        self.residue_definitions = residue_definitions
        self.bonds = bonds

        self.entity_id = entity_id
        self.mol_type = mol_type
        self.description = description
        self.formula_weight = formula_weight
        self.poly_type = poly_type
        self.nstd_linkage = nstd_linkage
        self.nstd_monomer = nstd_monomer
        self.sequence = sequence
        self.strand_id = strand_id

        # build topology
        self.topology = None

        # store coordinates
        self.coordinates = None

        # build kd tree
        self._kd_tree = None

    # atom selection DSL
    def select(self, expression):
        # atom/bond/angle/torsion/residue indices
        raise NotImplementedError

    def find_neighbors(self, idx, r):
        # find all atoms within distance r from atoms in idx
        raise NotImplementedError

    def find_k_neighbors(self, idx, k):
        # find k nearest neighbors to atoms in idx
        raise NotImplementedError

    def find_atoms_in_ball(self, x, r):
        # find all atoms within distance r from position x
        raise NotImplementedError

    def find_atom_pairs(self, r):
        # find all atom pairs separated by at most r
        raise NotImplementedError

    def sparse_distance_matrix(self, r_max):
        # return sparse distance matrix with cutoff r_max
        raise NotImplementedError


class Topology:
    # bonds
    # internal coordinate specifications
    # chain structure
    # sequence
    pass


s = StateSet.from_rcsb("4v4g")

#%%