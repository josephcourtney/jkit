import importlib.resources
import json
from collections import defaultdict

import numpy as np
import scipy.sparse as ss


class AminoAcidData:
    BACKBONE_ATOM_OFFSET = 3  # Add a constant for magic number

    def __init__(self):
        self.data = self.load_amino_acid_data()
        self.residues_1 = self.data["residues_1"]
        self.residues_3 = self.data["residues_3"]
        self.three_to_one = self.data["three_to_one"]
        self.one_to_three = {v: k for k, v in self.three_to_one.items()}
        self.atom_names = self.data["atom_names"]
        self.chemical_shifts_basic = self.data["chemical_shifts_basic"]
        self.bonds = self.data["bonds"]
        self.bond_types = self.process_bond_types()
        self.atom_ambiguity = self.process_atom_ambiguity()
        self.adjacency_matrices = self.generate_adjacency_matrices()
        self.distance_matrices = self.generate_distance_matrices()
        self.sparse_adjacency_matrices = self.generate_sparse_adjacency_matrices()

    @staticmethod
    def load_amino_acid_data():
        with importlib.resources.files("jkit").joinpath("amino_acid_data.json").open() as f:
            return json.load(f)

    def process_bond_types(self):
        bond_types = defaultdict(lambda: "single")
        for aa, a1, a2, t in self.data["bond_types"]:
            bond_types[aa, a1, a2] = t
            bond_types[aa, a2, a1] = t
        return bond_types

    def process_atom_ambiguity(self):
        atom_ambiguity = defaultdict(lambda: defaultdict(lambda: "Z"))
        for aa, atoms in self.data["atom_ambiguity"].items():
            atom_ambiguity[aa] = defaultdict(lambda: "Z", atoms)
        return atom_ambiguity

    def generate_adjacency_matrices(self):
        adjacency_matrices = {}
        for aa, bond_dict in self.bonds.items():
            n_atoms = len(bond_dict.keys())
            adjacency_matrices[aa] = np.zeros((n_atoms, n_atoms))
            for atm1, bonded_atoms in bond_dict.items():
                i = self.atom_names[aa].index(atm1)
                for atm2 in bonded_atoms:
                    j = self.atom_names[aa].index(atm2)
                    adjacency_matrices[aa][i, j] = 1
        return adjacency_matrices

    def generate_distance_matrices(self):
        distance_matrices = {}
        for aa, bond_dict in self.bonds.items():
            n_atoms = len(bond_dict.keys())
            distance_matrices[aa] = np.zeros((n_atoms, n_atoms))
            i = 1
            while np.any(distance_matrices[aa] == 0):
                idx = np.where(
                    np.logical_and(
                        distance_matrices[aa] == 0,
                        np.linalg.matrix_power(self.adjacency_matrices[aa], i) != 0,
                    )
                )
                distance_matrices[aa][idx] = i
                i += 1
            np.fill_diagonal(distance_matrices[aa], 0)
        return distance_matrices

    def generate_sparse_adjacency_matrices(self):
        sparse_adjacency_matrices = {}
        for aa, bond_dict in self.bonds.items():
            n_atoms = len(bond_dict.keys())
            sparse_adjacency_matrices[aa] = ss.dok_matrix((n_atoms, n_atoms))
            for atm1, bonded_atoms in bond_dict.items():
                i = self.atom_names[aa].index(atm1)
                for atm2 in bonded_atoms:
                    j = self.atom_names[aa].index(atm2)
                    sparse_adjacency_matrices[aa][i, j] = 1
        return sparse_adjacency_matrices


amino_acid_data = AminoAcidData()


def atoms_within_bond_distance(res_type, atom_name, n):
    if res_type in amino_acid_data.one_to_three:
        res_type = amino_acid_data.one_to_three[res_type]
    if res_type not in amino_acid_data.distance_matrices:
        return []
    if atom_name not in amino_acid_data.atom_names[res_type]:
        return []
    d = amino_acid_data.distance_matrices[res_type][amino_acid_data.atom_names[res_type].index(atom_name)]
    atoms = []
    for idx, e in enumerate(d):
        if isinstance(n, int) and e == n:
            atoms.append((res_type, amino_acid_data.atom_names[res_type][idx]))
            continue
        if isinstance(n, list | tuple) and e in n:
            atoms.append((res_type, amino_acid_data.atom_names[res_type][idx]))
            continue
    return atoms


def atom_to_index(sequence, res_num, atom_name):
    try:
        n = np.cumsum([len(amino_acid_data.atom_names[amino_acid_data.one_to_three[r]]) for r in sequence])
        return n[res_num - 1] + amino_acid_data.atom_names[
            amino_acid_data.one_to_three[sequence[res_num - 1]]
        ].index(atom_name)
    except (IndexError, KeyError, TypeError) as e:
        msg = f"Cannot determine index of atom atom {atom_name} for residue {res_num} in the given sequence"
        raise IndexError(msg) from e


def index_to_atom(sequence, index):
    try:
        n = np.cumsum([len(amino_acid_data.atom_names[amino_acid_data.one_to_three[r]]) for r in sequence])
        idx = np.where(index < n)[0][0]
        r = sequence[idx]
        a_idx = index - (0 if idx == 0 else n[idx - 1])
        return idx, amino_acid_data.atom_names[amino_acid_data.one_to_three[r]][a_idx]
    except (KeyError, IndexError, TypeError) as e:
        msg = f"Cannot locate the atom at index {index} in the given sequence"
        raise IndexError(msg) from e


def sequence_to_adjacency_matrix(sequence):
    n = np.cumsum([len(amino_acid_data.atom_names[amino_acid_data.one_to_three[r]]) for r in sequence])
    n[0], n[1:] = 0, n[:-1]
    mat = ss.block_diag(
        [amino_acid_data.sparse_adjacency_matrices[amino_acid_data.one_to_three[r]] for r in sequence],
        format="csr",
    )
    for i in range(len(sequence) - 1):
        mat[n[i], n[i + 1] + AminoAcidData.BACKBONE_ATOM_OFFSET] = -1
        mat[n[i + 1] + AminoAcidData.BACKBONE_ATOM_OFFSET, n[i]] = -1
    return mat
