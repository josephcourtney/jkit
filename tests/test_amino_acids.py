from collections import defaultdict

from jkit.amino_acids import (
    AminoAcidData,
    atom_to_index,
    atoms_within_bond_distance,
    index_to_atom,
    sequence_to_adjacency_matrix,
)

# Create an instance of AminoAcidData
amino_acid_data = AminoAcidData()


def test_residue_mappings():
    assert len(amino_acid_data.residues_1) == len(amino_acid_data.residues_3)
    assert all(
        amino_acid_data.three_to_one[res] in amino_acid_data.residues_1 for res in amino_acid_data.residues_3
    )
    assert all(
        amino_acid_data.one_to_three[one] in amino_acid_data.residues_3 for one in amino_acid_data.residues_1
    )


def test_bond_symmetry():
    for aa in amino_acid_data.bonds:
        for atm1 in amino_acid_data.bonds[aa]:
            for atm2 in amino_acid_data.bonds[aa][atm1]:
                assert atm2 in amino_acid_data.bonds[aa]
                assert atm1 in amino_acid_data.bonds[aa][atm2]


def test_adjacency_matrices():
    for matrix in amino_acid_data.adjacency_matrices.values():
        assert matrix.shape[0] == matrix.shape[1]
        assert (matrix == matrix.T).all()


def test_atoms_within_bond_distance():
    atoms = atoms_within_bond_distance("GLY", "CA", 1)
    assert len(atoms) > 0
    assert all(atom[0] == "GLY" for atom in atoms)


def test_sequence_to_adjacency_matrix():
    sequence = ["G", "A", "S"]
    assert all(residue in amino_acid_data.one_to_three for residue in sequence), (
        "Residue not found in one_to_three dictionary"
    )
    matrix = sequence_to_adjacency_matrix(sequence)
    assert matrix is not None


def test_bonds():
    # All atoms have full valence:
    for aa in amino_acid_data.bonds:
        for atm1 in amino_acid_data.bonds[aa]:
            if atm1[0] == "H":
                assert len(amino_acid_data.bonds[aa][atm1]) == 1
            elif atm1[0] == "O":  # exception for double-bonded oxygens
                assert (
                    len(amino_acid_data.bonds[aa][atm1]) == 2
                    or (aa, atm1)
                    in {
                        ("ASP", "OD1"),
                        ("ASN", "OD1"),
                        ("GLU", "OE1"),
                        ("GLN", "OE1"),
                    }
                    or atm1 == "O"
                )
            elif atm1[0] == "S":
                assert len(amino_acid_data.bonds[aa][atm1]) == 2
            elif atm1[0] == "N":
                assert (
                    len(amino_acid_data.bonds[aa][atm1]) == 3 or atm1 == "N" or (aa == "LYS" and atm1 == "NZ")
                )
            elif atm1[0] == "C":
                assert (
                    len(amino_acid_data.bonds[aa][atm1]) != 4
                    or atm1 != "C"
                    or (aa, atm1)
                    in {
                        # Methyls with indistinguishable protons
                        ("ALA", "CB"),
                        ("ILE", "CD1"),
                        ("ILE", "CG2"),
                        ("LEU", "CD1"),
                        ("LEU", "CD2"),
                        ("MET", "CE"),
                        ("THR", "CG2"),
                        ("VAL", "CG1"),
                        ("VAL", "CG2"),
                        # Aromatic carbons
                        ("PHE", "CG"),
                        ("PHE", "CD1"),
                        ("PHE", "CD2"),
                        ("PHE", "CE1"),
                        ("PHE", "CE2"),
                        ("PHE", "CZ"),
                        ("TYR", "CG"),
                        ("TYR", "CD1"),
                        ("TYR", "CD2"),
                        ("TYR", "CE1"),
                        ("TYR", "CE2"),
                        ("TYR", "CZ"),
                        ("TRP", "CD2"),
                        ("TRP", "CE2"),
                        ("TRP", "CE3"),
                        ("TRP", "CZ2"),
                        ("TRP", "CZ3"),
                        ("TRP", "CH2"),
                        ("TRP", "CG"),
                        ("TRP", "CD1"),
                        ("HIS", "CG"),
                        ("HIS", "CD2"),
                        ("HIS", "CE1"),
                        # Sidechain carboxyls
                        ("ASP", "CG"),
                        ("GLU", "CD"),
                        # Sidechain amides
                        ("ASN", "CG"),
                        ("GLN", "CD"),
                        # Arginine guanidinium
                        ("ARG", "CZ"),
                    }
                )


# Additional Tests


def test_load_amino_acid_data():
    data = amino_acid_data.load_amino_acid_data()
    assert isinstance(data, dict), "Data should be loaded as a dictionary"
    assert "residues_1" in data, "residues_1 key should be present in the data"


def test_process_bond_types():
    bond_types = amino_acid_data.process_bond_types()
    assert isinstance(bond_types, defaultdict), "Bond types should be a defaultdict"
    assert ("CYS", "C", "O") in bond_types, "Bond type for ('CYS', 'C', 'O') should be present"
    assert bond_types["CYS", "C", "O"] == "delocalized", (
        "Bond type for ('CYS', 'C', 'O') should be 'delocalized'"
    )


def test_process_atom_ambiguity():
    atom_ambiguity = amino_acid_data.process_atom_ambiguity()
    assert isinstance(atom_ambiguity, defaultdict), "Atom ambiguity should be a defaultdict"
    assert "CYS" in atom_ambiguity, "CYS should be present in atom ambiguity"
    assert atom_ambiguity["CYS"]["C"] == "C", "Atom ambiguity for CYS 'C' should be 'C'"


def test_generate_adjacency_matrices():
    adjacency_matrices = amino_acid_data.generate_adjacency_matrices()
    for aa, matrix in adjacency_matrices.items():
        assert matrix.shape[0] == len(amino_acid_data.atom_names[aa]), (
            "Adjacency matrix dimensions should match number of atoms"
        )
        assert (matrix == matrix.T).all(), "Adjacency matrix should be symmetric"


def test_generate_distance_matrices():
    distance_matrices = amino_acid_data.generate_distance_matrices()
    for aa, matrix in distance_matrices.items():
        assert matrix.shape[0] == len(amino_acid_data.atom_names[aa]), (
            "Distance matrix dimensions should match number of atoms"
        )
        assert (matrix.diagonal() == 0).all(), "Distance matrix diagonal should be zero"


def test_atom_to_index_edge_case():
    index = atom_to_index(["A", "G", "P"], 2, "N")
    assert index is not None, "Index should not be None for valid input"


def test_index_to_atom_edge_case():
    idx, atom = index_to_atom(["A", "G", "P"], 10)
    assert idx is not None, "Index should not be None for valid input"
    assert atom is not None, "Atom should not be None for valid input"


def test_invalid_atom_to_index():
    index = atom_to_index(["A", "G", "P"], 10, "N")
    assert index is None, "Index should be None for invalid input"


def test_invalid_index_to_atom():
    idx, atom = index_to_atom(["A", "G", "P"], 100)
    assert idx is None, "Index should be None for invalid input"
    assert atom is None, "Atom should be None for invalid input"


def test_invalid_atoms_within_bond_distance():
    atoms = atoms_within_bond_distance("ZZZ", "CA", 1)
    assert len(atoms) == 0, "No atoms should be returned for invalid residue type"
