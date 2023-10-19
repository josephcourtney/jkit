# Collection of useful information about amino acids
#%%
from collections import defaultdict

import numpy as np
import scipy.sparse as ss

residues_1 = ['C', 'D', 'S', 'Q', 'K', 'P', 'T', 'F', 'A', 'H', 'G', 'I', 'E', 'L', 'R', 'W', 'V', 'N', 'Y', 'M']
residues_3 = ['CYS', 'ASP', 'SER', 'GLN', 'LYS', 'PRO', 'THR', 'PHE', 'ALA', 'HIS', 'GLY', 'ILE', 'GLU', 'LEU', 'ARG', 'TRP', 'VAL', 'ASN', 'TYR', 'MET']

three_to_one = {
    'CYS': 'C',
    'ASP': 'D',
    'SEP': 'S',
    'SER': 'S',
    'GLN': 'Q',
    'LYS': 'K',
    'PRO': 'P',
    'THR': 'T',
    'PHE': 'F',
    'ALA': 'A',
    'HIS': 'H',
    'GLY': 'G',
    'ILE': 'I',
    'GLU': 'E',
    'LEU': 'L',
    'ARG': 'R',
    'TRP': 'W',
    'VAL': 'V',
    'ASN': 'N',
    'TYR': 'Y',
    'MET': 'M'
}
one_to_three = {v:k for k,v in three_to_one.items()}

atom_names = {
    'ALA': ['C', 'CA', 'CB', 'H', 'HA', 'HB1', 'HB2', 'HB3', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CD', 'CG', 'CZ', 'H', 'HA', 'HB2', 'HB3', 'HD2', 'HD3', 'HE', 'HG2', 'HG3', 'HH11', 'HH12', 'HH21', 'HH22', 'N', 'NE', 'NH1', 'NH2', 'O'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'H', 'HA', 'HB2', 'HB3', 'HD2', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'H', 'HA', 'HB2', 'HB3', 'HD21', 'HD22', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'H', 'HA', 'HB2', 'HB3', 'HG', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CD', 'CG', 'H', 'HA', 'HB2', 'HB3', 'HE2', 'HG2', 'HG3', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CD', 'CG', 'H', 'HA', 'HB2', 'HB3', 'HE21', 'HE22', 'HG2', 'HG3', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'H', 'HA2', 'HA3', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CD2', 'CE1', 'CG', 'H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CD1', 'CG1', 'CG2', 'H', 'HA', 'HB', 'HD11', 'HD12', 'HD13', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CD1', 'CD2', 'CG', 'H', 'HA', 'HB2', 'HB3', 'HD11', 'HD12', 'HD13', 'HD21', 'HD22', 'HD23', 'HG', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CD', 'CE', 'CG', 'H', 'HA', 'HB2', 'HB3', 'HD2', 'HD3', 'HE2', 'HE3', 'HG2', 'HG3', 'HZ1', 'HZ2', 'HZ3', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CE', 'CG', 'H', 'HA', 'HB2', 'HB3', 'HE1', 'HE2', 'HE3', 'HG2', 'HG3', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CD', 'CG', 'HA', 'HB2', 'HB3', 'HD2', 'HD3', 'HG2', 'HG3', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'H', 'HA', 'HB2', 'HB3', 'HG', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'H', 'HA', 'HB', 'HG1', 'HG21', 'HG22', 'HG23', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CD1', 'CD2', 'CE2', 'CE3', 'CG', 'CH2', 'CZ2', 'CZ3', 'H', 'HA', 'HB2', 'HB3', 'HD1', 'HE1', 'HE3', 'HH2', 'HZ2', 'HZ3', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CD1', 'CD2', 'CE1', 'CE2', 'CG', 'CZ', 'H', 'HA', 'HB2', 'HB3', 'HD1', 'HD2', 'HE1', 'HE2', 'HH', 'N', 'O', 'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'H', 'HA', 'HB', 'HG11', 'HG12', 'HG13', 'HG21', 'HG22', 'HG23', 'N', 'O'],
}

chemical_shifts_basic = {
    'CYS':
        {
            'C': (174.92, 2.04),
            'CA': (58.16, 3.4),
            'CB': (33.0, 6.31),
            'H': (8.38, 0.68),
            'HA': (4.65, 0.55),
            'HB2': (2.95, 0.44),
            'HB3': (2.89, 0.45),
            'HG': (2.01, 1.16),
            'N': (120.09, 4.64)
        },
    'ASP':
        {
            'C': (176.44, 1.73),
            'CA': (54.7, 2.04),
            'CB': (40.86, 1.63),
            'CG': (179.31, 1.83),
            'H': (8.3, 0.57),
            'HA': (4.58, 0.31),
            'HB2': (2.71, 0.26),
            'HB3': (2.66, 0.27),
            'HD2': (6.06, 1.87),
            'N': (120.67, 3.8)
        },
    'SER':
        {
            'C': (174.66, 1.74),
            'CA': (58.75, 2.08),
            'CB': (63.78, 1.53),
            'H': (8.28, 0.58),
            'HA': (4.47, 0.4),
            'HB2': (3.87, 0.25),
            'HB3': (3.85, 0.27),
            'HG': (5.4, 1.02),
            'N': (116.27, 3.5)
        },
    'GLN':
        {
            'C': (176.36, 1.93),
            'CA': (56.61, 2.12),
            'CB': (29.15, 1.82),
            'CD': (179.72, 1.23),
            'CG': (33.77, 1.13),
            'H': (8.22, 0.58),
            'HA': (4.26, 0.43),
            'HB2': (2.05, 0.25),
            'HB3': (2.02, 0.26),
            'HE21': (7.21, 0.44),
            'HE22': (7.04, 0.43),
            'HG2': (2.31, 0.26),
            'HG3': (2.29, 0.28),
            'N': (119.89, 3.54),
            'NE2': (111.86, 1.69)
        },
    'LYS':
        {
            'C': (176.71, 1.93),
            'CA': (56.98, 2.18),
            'CB': (32.77, 1.77),
            'CD': (28.95, 1.11),
            'CE': (41.88, 0.89),
            'CG': (24.89, 1.15),
            'H': (8.18, 0.59),
            'HA': (4.26, 0.43),
            'HB2': (1.78, 0.25),
            'HB3': (1.75, 0.27),
            'HD2': (1.6, 0.21),
            'HD3': (1.6, 0.22),
            'HE2': (2.91, 0.19),
            'HE3': (2.91, 0.2),
            'HG2': (1.37, 0.25),
            'HG3': (1.35, 0.27),
            'HZ1': (7.4, 0.66),
            'HZ2': (7.4, 0.66),
            'HZ3': (7.4, 0.66),
            'N': (121.03, 3.7),
            'NZ': (33.14, 1.75)
        },
    'ILE':
        {
            'C': (175.92, 1.91),
            'CA': (61.67, 2.69),
            'CB': (38.57, 2.0),
            'CD1': (13.39, 1.67),
            'CG1': (27.73, 1.72),
            'CG2': (17.52, 1.35),
            'H': (8.27, 0.68),
            'HA': (4.16, 0.56),
            'HB': (1.78, 0.29),
            'HD11': (0.68, 0.29),
            'HD12': (0.68, 0.29),
            'HD13': (0.68, 0.29),
            'HG12': (1.27, 0.4),
            'HG13': (1.19, 0.41),
            'HG21': (0.78, 0.27),
            'HG22': (0.78, 0.27),
            'HG23': (0.78, 0.27),
            'N': (121.41, 4.25)
        },
    'PRO':
        {
            'C': (176.76, 1.5),
            'CA': (63.34, 1.56),
            'CB': (31.83, 1.2),
            'CD': (50.32, 1.06),
            'CG': (27.19, 1.12),
            'HA': (4.39, 0.33),
            'HB2': (2.07, 0.35),
            'HB3': (2.0, 0.36),
            'HD2': (3.65, 0.35),
            'HD3': (3.61, 0.39),
            'HG2': (1.92, 0.31),
            'HG3': (1.9, 0.33),
            'N': (134.55, 6.29)
        },
    'THR':
        {
            'C': (174.57, 1.73),
            'CA': (62.26, 2.6),
            'CB': (69.7, 1.78),
            'CG2': (21.54, 1.11),
            'H': (8.24, 0.62),
            'HA': (4.45, 0.47),
            'HB': (4.16, 0.32),
            'HG1': (5.17, 1.14),
            'HG21': (1.14, 0.22),
            'HG22': (1.14, 0.22),
            'HG23': (1.14, 0.22),
            'N': (115.37, 4.72)
        },
    'PHE':
        {
            'C': (175.47, 1.98),
            'CA': (58.13, 2.59),
            'CB': (39.94, 2.06),
            'CD1': (131.58, 1.22),
            'CD2': (131.59, 1.22),
            'CE1': (130.73, 1.32),
            'CE2': (130.76, 1.2),
            'CG': (138.43, 2.87),
            'CZ': (129.21, 1.48),
            'H': (8.34, 0.71),
            'HA': (4.61, 0.56),
            'HB2': (2.99, 0.37),
            'HB3': (2.94, 0.39),
            'HD1': (7.06, 0.31),
            'HD2': (7.06, 0.31),
            'HE1': (7.08, 0.31),
            'HE2': (7.08, 0.31),
            'HZ': (7.0, 0.42),
            'N': (120.37, 4.14)
        },
    'ASN':
        {
            'C': (175.3, 1.78),
            'CA': (53.56, 1.88),
            'CB': (38.69, 1.67),
            'CG': (176.77, 1.4),
            'H': (8.32, 0.62),
            'HA': (4.66, 0.36),
            'HB2': (2.8, 0.32),
            'HB3': (2.75, 0.33),
            'HD21': (7.32, 0.49),
            'HD22': (7.15, 0.5),
            'N': (118.91, 3.94),
            'ND2': (112.76, 2.29)
        },
    'GLY':
        {
            'C': (173.9, 1.86),
            'CA': (45.36, 1.32),
            'H': (8.33, 0.63),
            'HA2': (3.96, 0.37),
            'HA3': (3.9, 0.37),
            'N': (109.6, 3.72)
        },
    'HIS':
        {
            'C': (175.25, 1.95),
            'CA': (56.51, 2.32),
            'CB': (30.22, 2.1),
            'CD2': (120.42, 3.43),
            'CE1': (137.64, 2.29),
            'CG': (131.88, 3.32),
            'H': (8.25, 0.68),
            'HA': (4.6, 0.43),
            'HB2': (3.1, 0.35),
            'HB3': (3.05, 0.38),
            'HD1': (8.57, 2.49),
            'HD2': (7.0, 0.42),
            'HE1': (7.96, 0.48),
            'HE2': (9.6, 2.42),
            'N': (119.71, 4.03),
            'ND1': (193.04, 18.29),
            'NE2': (185.02, 16.71)
        },
    'LEU':
        {
            'C': (177.06, 1.95),
            'CA': (55.69, 2.13),
            'CB': (42.25, 1.86),
            'CD1': (24.66, 1.6),
            'CD2': (24.07, 1.7),
            'CG': (26.77, 1.11),
            'H': (8.22, 0.64),
            'HA': (4.3, 0.46),
            'HB2': (1.61, 0.34),
            'HB3': (1.52, 0.36),
            'HD11': (0.75, 0.28),
            'HD12': (0.75, 0.28),
            'HD13': (0.75, 0.28),
            'HD21': (0.73, 0.28),
            'HD22': (0.73, 0.28),
            'HD23': (0.73, 0.28),
            'HG': (1.51, 0.33),
            'N': (121.82, 3.86)
        },
    'ARG':
        {
            'C': (176.47, 2.01),
            'CA': (56.81, 2.31),
            'CB': (30.64, 1.82),
            'CD': (43.14, 0.94),
            'CG': (27.21, 1.22),
            'CZ': (159.98, 3.72),
            'H': (8.23, 0.61),
            'HA': (4.29, 0.46),
            'HB2': (1.79, 0.26),
            'HB3': (1.76, 0.28),
            'HD2': (3.12, 0.23),
            'HD3': (3.1, 0.25),
            'HE': (7.36, 0.6),
            'HG2': (1.57, 0.27),
            'HG3': (1.54, 0.28),
            'HH11': (6.89, 0.47),
            'HH12': (6.86, 0.49),
            'HH21': (6.81, 0.49),
            'HH22': (6.82, 0.5),
            'N': (120.79, 3.64),
            'NE': (84.58, 1.61),
            'NH1': (74.28, 5.09),
            'NH2': (72.7, 2.69)
        },
    'TRP':
        {
            'C': (176.22, 1.99),
            'CA': (57.73, 2.57),
            'CB': (29.98, 2.01),
            'CD1': (126.55, 1.83),
            'CD2': (127.81, 1.63),
            'CE2': (138.01, 7.34),
            'CE3': (120.45, 1.82),
            'CG': (110.38, 7.48),
            'CH2': (123.82, 1.57),
            'CZ2': (114.27, 1.46),
            'CZ3': (121.37, 1.61),
            'H': (8.27, 0.77),
            'HA': (4.66, 0.52),
            'HB2': (3.19, 0.35),
            'HB3': (3.12, 0.37),
            'HD1': (7.14, 0.35),
            'HE1': (10.08, 0.64),
            'HE3': (7.32, 0.41),
            'HH2': (6.98, 0.37),
            'HZ2': (7.28, 0.32),
            'HZ3': (6.87, 0.38),
            'N': (121.58, 4.06),
            'NE1': (129.29, 2.07)
        },
    'ALA':
        {
            'C': (177.81, 2.08),
            'CA': (53.18, 1.95),
            'CB': (18.97, 1.79),
            'H': (8.19, 0.59),
            'HA': (4.24, 0.43),
            'HB1': (1.36, 0.25),
            'HB2': (1.36, 0.25),
            'HB3': (1.36, 0.25),
            'N': (123.28, 3.48)
        },
    'VAL':
        {
            'C': (175.7, 1.87),
            'CA': (62.57, 2.85),
            'CB': (32.7, 1.78),
            'CG1': (21.5, 1.37),
            'CG2': (21.28, 1.54),
            'H': (8.28, 0.67),
            'HA': (4.16, 0.57),
            'HB': (1.98, 0.31),
            'HG11': (0.83, 0.26),
            'HG12': (0.83, 0.26),
            'HG13': (0.83, 0.26),
            'HG21': (0.8, 0.28),
            'HG22': (0.8, 0.28),
            'HG23': (0.8, 0.28),
            'N': (121.1, 4.46)
        },
    'GLU':
        {
            'C': (176.93, 1.92),
            'CA': (57.35, 2.08),
            'CB': (29.95, 1.7),
            'CD': (182.14, 3.38),
            'CG': (36.09, 1.21),
            'H': (8.33, 0.59),
            'HA': (4.24, 0.4),
            'HB2': (2.02, 0.21),
            'HB3': (2.0, 0.21),
            'HE2': (2.82, 0.1),
            'HG2': (2.27, 0.21),
            'HG3': (2.25, 0.21),
            'N': (120.7, 3.45)
        },
    'TYR':
        {
            'C': (175.46, 1.97),
            'CA': (58.18, 2.51),
            'CB': (39.27, 2.16),
            'CD1': (132.73, 1.35),
            'CD2': (132.69, 1.51),
            'CE1': (117.94, 1.29),
            'CE2': (117.9, 1.26),
            'CG': (129.85, 4.24),
            'CZ': (156.5, 4.61),
            'H': (8.3, 0.73),
            'HA': (4.6, 0.56),
            'HB2': (2.9, 0.38),
            'HB3': (2.84, 0.39),
            'HD1': (6.93, 0.3),
            'HD2': (6.93, 0.3),
            'HE1': (6.7, 0.22),
            'HE2': (6.7, 0.23),
            'HH': (9.13, 1.63),
            'N': (120.49, 4.24)
        },
    'MET':
        {
            'C': (176.23, 2.08),
            'CA': (56.15, 2.22),
            'CB': (32.93, 2.2),
            'CE': (17.1, 1.71),
            'CG': (32.02, 1.3),
            'H': (8.25, 0.59),
            'HA': (4.39, 0.46),
            'HB2': (2.02, 0.33),
            'HB3': (1.99, 0.34),
            'HE1': (1.89, 0.41),
            'HE2': (1.89, 0.41),
            'HE3': (1.89, 0.41),
            'HG2': (2.42, 0.35),
            'HG3': (2.39, 0.38),
            'N': (120.1, 3.5)}
}


bonds = {
    'CYS':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','SG','HB2','HB3'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HG': ['SG'],
            'SG': ['CB','HG']
        },
    'ASP':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CG': ['CB','OD1','OD2'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD2': ['OD2'],
            'OD1': ['CG'],
            'OD2': ['CG','HD2']
        },
    'SER':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','OG','HB2','HB3'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HG': ['OG'],
            'OG': ['CB','HG']
        },
    'GLN':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD': ['OE1','NE2','CG'],
            'CG': ['CB','HG2','HG3','CD'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HE21': ['NE2'],
            'HE22': ['NE2'],
            'HG2': ['CG'],
            'HG3': ['CG'],
            'NE2': ['HE22','HE21','CD'],
            'OE1': ['CD']
        },
    'LYS':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD': ['CG','CE','HD2','HD3'],
            'CE': ['CD','NZ','HE2','HE3'],
            'CG': ['CB','CD','HG2','HG3'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD2': ['CD'],
            'HD3': ['CD'],
            'HE2': ['CE'],
            'HE3': ['CE'],
            'HG2': ['CG'],
            'HG3': ['CG'],
            'HZ1': ['NZ'],
            'HZ2': ['NZ'],
            'HZ3': ['NZ'],
            'NZ': ['CE','HZ1','HZ2','HZ3']
        },
    'ILE':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG1','CG2','HB'],
            'CD1': ['CG1','HD11','HD12','HD13'],
            'CG1': ['CB','CD1','HG12','HG13'],
            'CG2': ['CB','HG21','HG22','HG23'],
            'H': ['N'],
            'HA': ['CA'],
            'HB': ['CB'],
            'HD11': ['CD1'],
            'HD12': ['CD1'],
            'HD13': ['CD1'],
            'HG12': ['CG1'],
            'HG13': ['CG1'],
            'HG21': ['CG2'],
            'HG22': ['CG2'],
            'HG23': ['CG2'],
        },
    'PRO':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['CD','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD': ['N','CG','HD2','HD3'],
            'CG': ['CB','CD','HG2','HG3'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD2': ['CD'],
            'HD3': ['CD'],
            'HG2': ['CG'],
            'HG3': ['CG'],
        },
    'THR':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','OG1','CG2','HB'],
            'CG2': ['CB','HG21','HG22','HG23'],
            'H': ['N'],
            'HA': ['CA'],
            'HB': ['CB'],
            'HG1': ['OG1'],
            'HG21': ['CG2'],
            'HG22': ['CG2'],
            'HG23': ['CG2'],
            'OG1': ['CB','HG1']
        },
    'PHE':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD1': ['CG','CE1','HD1'],
            'CD2': ['CG','CE2','HD2'],
            'CE1': ['CD1','CZ','HE1'],
            'CE2': ['CD2','CZ','HE2'],
            'CG': ['CD1','CD2','CB'],
            'CZ': ['CE1','CE2','HZ'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD1': ['CD1'],
            'HD2': ['CD2'],
            'HE1': ['CE1'],
            'HE2': ['CE2'],
            'HZ': ['CZ'],
        },
    'ASN':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CG': ['CB','ND2','OD1'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD21': ['ND2'],
            'HD22': ['ND2'],
            'ND2': ['CG','HD21','HD22'],
            'OD1': ['CG']
        },
    'GLY':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA2','HA3','C','N'],
            'N': ['H','CA'],
            'H': ['N'],
            'HA2': ['CA'],
            'HA3': ['CA'],
        },
    'HIS':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD2': ['CG','NE2','HD2'],
            'CE1': ['ND1','NE2','HE1'],
            'CG': ['CB','ND1','CD2'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD1': ['ND1'],
            'HD2': ['CD2'],
            'HE1': ['CE1'],
            'HE2': ['NE2'],
            'ND1': ['CG','CE1','HD1'],
            'NE2': ['CD2','CE1','HE2']
        },
    'LEU':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD1': ['CG','HD11','HD12','HD13'],
            'CD2': ['CG','HD21','HD22','HD23'],
            'CG': ['CB','CD1','CD2','HG'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD11': ['CD1'],
            'HD12': ['CD1'],
            'HD13': ['CD1'],
            'HD21': ['CD2'],
            'HD22': ['CD2'],
            'HD23': ['CD2'],
            'HG': ['CG'],
        },
    'ARG':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD': ['CG','NE','HD2','HD3'],
            'CG': ['CB','CD','HG2','HG3'],
            'CZ': ['NE','NH2','NH1'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD2': ['CD'],
            'HD3': ['CD'],
            'HE': ['NE'],
            'HG2': ['CG'],
            'HG3': ['CG'],
            'HH11': ['NH1'],
            'HH12': ['NH1'],
            'HH21': ['NH2'],
            'HH22': ['NH2'],
            'NE': ['CD','CZ','HE'],
            'NH1': ['CZ','HH11','HH12'],
            'NH2': ['CZ','HH21','HH22']
        },
    'TRP':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD1': ['CG','NE1','HD1'],
            'CD2': ['CG','CE2','CE3'],
            'CE2': ['CD2','NE1','CZ2'],
            'CE3': ['CD2','CZ3','HE3'],
            'CG': ['CB','CD1','CD2'],
            'CH2': ['CZ2','CZ3','HH2'],
            'CZ2': ['CE2','CH2','HZ2'],
            'CZ3': ['CE3','CH2','HZ3'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD1': ['CD1'],
            'HE1': ['NE1'],
            'HE3': ['CE3'],
            'HH2': ['CH2'],
            'HZ2': ['CZ2'],
            'HZ3': ['CZ3'],
            'NE1': ['CD1','CE2','HE1']
        },
    'ALA':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','HB1','HB2','HB3'],
            'H': ['N'],
            'HA': ['CA'],
            'HB1': ['CB'],
            'HB2': ['CB'],
            'HB3': ['CB'],
        },
    'VAL':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG1','CG2','HB'],
            'CG1': ['CB','HG11','HG12','HG13'],
            'CG2': ['CB','HG21','HG22','HG23'],
            'H': ['N'],
            'HA': ['CA'],
            'HB': ['CB'],
            'HG11': ['CG1'],
            'HG12': ['CG1'],
            'HG13': ['CG1'],
            'HG21': ['CG2'],
            'HG22': ['CG2'],
            'HG23': ['CG2'],
        },
    'GLU':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD': ['CG','OE1','OE2'],
            'CG': ['CB','CD','HG2','HG3'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HE2': ['OE2'],
            'HG2': ['CG'],
            'HG3': ['CG'],
            'OE1': ['CD'],
            'OE2': ['CD', 'HE2']
        },
    'TYR':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CD1': ['CG','CE1','HD1'],
            'CD2': ['CG','CE2','HD2'],
            'CE1': ['CD1','CZ','HE1'],
            'CE2': ['CD2','CZ','HE2'],
            'CG': ['CD1','CD2','CB'],
            'CZ': ['CE1','CE2','OH'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HD1': ['CD1'],
            'HD2': ['CD2'],
            'HE1': ['CE1'],
            'HE2': ['CE2'],
            'HH': ['OH'],
            'OH': ['CZ','HH']
        },
    'MET':
        {
            'C': ['CA', 'O'],
            'O': ['C'],
            'CA': ['HA','C','N','CB'],
            'N': ['H','CA'],
            'CB': ['CA','CG','HB2','HB3'],
            'CE': ['SD','HE1','HE2','HE3'],
            'CG': ['CB','SD','HG2','HG3'],
            'H': ['N'],
            'HA': ['CA'],
            'HB2': ['CB'],
            'HB3': ['CB'],
            'HE1': ['CE'],
            'HE2': ['CE'],
            'HE3': ['CE'],
            'HG2': ['CG'],
            'HG3': ['CG'],
            'SD': ['CG', 'CE']
        }
}


bond_types = defaultdict(lambda : 'single')
for aa in residues_3:
    bond_types[(aa, 'C', 'O')] = 'delocalized'
    bond_types[(aa, 'C', 'N')] = 'delocalized'
bond_types[('ARG', 'NE', 'CZ')] = 'delocalized'
bond_types[('ARG', 'CZ', 'NH1')] = 'delocalized'
bond_types[('ARG', 'CZ', 'NH2')] = 'delocalized'
bond_types[('ASN', 'CG', 'OD1')] = 'double'
bond_types[('ASP', 'CG', 'OD1')] = 'double'
bond_types[('GLN', 'CD', 'OE1')] = 'double'
bond_types[('GLU', 'CD', 'OE1')] = 'double'
bond_types[('HIS', 'CG', 'ND1')] = 'aromatic'
bond_types[('HIS', 'CD2', 'CG')] = 'aromatic'
bond_types[('HIS', 'CE1', 'ND1')] = 'aromatic'
bond_types[('HIS', 'CD2', 'NE2')] = 'aromatic'
bond_types[('HIS', 'CE1', 'NE2')] = 'aromatic'
bond_types[('PHE', 'CD1', 'CG')] = 'aromatic'
bond_types[('PHE', 'CD2', 'CG')] = 'aromatic'
bond_types[('PHE', 'CD1', 'CE1')] = 'aromatic'
bond_types[('PHE', 'CD2', 'CE2')] = 'aromatic'
bond_types[('PHE', 'CE1', 'CZ')] = 'aromatic'
bond_types[('PHE', 'CE2', 'CZ')] = 'aromatic'
bond_types[('TRP', 'CD1', 'CG')] = 'aromatic'
bond_types[('TRP', 'CD1', 'NE1')] = 'aromatic'
bond_types[('TRP', 'CD2', 'CG')] = 'aromatic'
bond_types[('TRP', 'CD2', 'CE2')] = 'aromatic'
bond_types[('TRP', 'CD2', 'CE3')] = 'aromatic'
bond_types[('TRP', 'CE2', 'CZ2')] = 'aromatic'
bond_types[('TRP', 'CE2', 'NE1')] = 'aromatic'
bond_types[('TRP', 'CE3', 'CZ3')] = 'aromatic'
bond_types[('TRP', 'CH2', 'CZ2')] = 'aromatic'
bond_types[('TRP', 'CH2', 'CZ3')] = 'aromatic'
bond_types[('TYR', 'CD1', 'CG')] = 'aromatic'
bond_types[('TYR', 'CD2', 'CG')] = 'aromatic'
bond_types[('TYR', 'CD1', 'CE1')] = 'aromatic'
bond_types[('TYR', 'CD2', 'CE2')] = 'aromatic'
bond_types[('TYR', 'CE1', 'CZ')] = 'aromatic'
bond_types[('TYR', 'CE2', 'CZ')] = 'aromatic'
for (aa, a1, a2), t in list(bond_types.items()):
    bond_types[(aa, a2, a1)] = t



_atom_ambiguity = {
    'CYS': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HG': 'HG',
        'SG': 'SG',
    },
    'ASP': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CG': 'CG',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD2': 'HD2',
        'OD1': 'OD1',
        'OD2': 'OD2',
    },
    'SER': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HG': 'HG',
        'OG': 'OG',
    },
    'GLN': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD': 'CD',
        'CG': 'CG',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HE21': 'HE2*',
        'HE22': 'HE2*',
        'HG2': 'HG*',
        'HG3': 'HG*',
        'NE2': 'NE2',
        'OE1': 'OE1',
    },
    'LYS': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD': 'CD',
        'CE': 'CE',
        'CG': 'CG',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD2': 'HD*',
        'HD3': 'HD*',
        'HE2': 'HE*',
        'HE3': 'HE*',
        'HG2': 'HG*',
        'HG3': 'HG*',
        'HZ1': 'HZ*',
        'HZ2': 'HZ*',
        'HZ3': 'HZ*',
        'NZ': 'NZ',
    },
    'ILE': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD1': 'CD1',
        'CG1': 'CG1',
        'CG2': 'CG2',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB': 'HB',
        'HD11': 'HD1*',
        'HD12': 'HD1*',
        'HD13': 'HD1*',
        'HG12': 'HG1*',
        'HG13': 'HG1*',
        'HG21': 'HG2*',
        'HG22': 'HG2*',
        'HG23': 'HG2*',
    },
    'PRO': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD': 'CD',
        'CG': 'CG',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD2': 'HD*',
        'HD3': 'HD*',
        'HG2': 'HG*',
        'HG3': 'HG*',
    },
    'THR': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CG2': 'CG2',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB': 'HB',
        'HG1': 'HG1',
        'HG21': 'HG2*',
        'HG22': 'HG2*',
        'HG23': 'HG2*',
        'OG1': 'OG1',
    },
    'PHE': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD1': 'CD*',
        'CD2': 'CD*',
        'CE1': 'CE*',
        'CE2': 'CE*',
        'CG': 'CG',
        'CZ': 'CZ',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD1': 'HD*',
        'HD2': 'HD*',
        'HE1': 'HE*',
        'HE2': 'HE*',
        'HZ': 'HZ',
    },
    'ASN': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CG': 'CG',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD21': 'HD2*',
        'HD22': 'HD2*',
        'ND2': 'ND2',
        'OD1': 'OD1',
    },
    'GLY': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA2': 'HA*',
        'HA3': 'HA*',
    },
    'HIS': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD2': 'CD2',
        'CE1': 'CE1',
        'CG': 'CG',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD1': 'HD1',
        'HD2': 'HD2',
        'HE1': 'HE1',
        'HE2': 'HE2',
        'ND1': 'ND1',
        'NE2': 'NE2',
    },
    'LEU': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD1': 'CD*',
        'CD2': 'CD*',
        'CG': 'CG',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD11': 'HD*',
        'HD12': 'HD*',
        'HD13': 'HD*',
        'HD21': 'HD*',
        'HD22': 'HD*',
        'HD23': 'HD*',
        'HG': 'HG',
    },
    'ARG': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD': 'CD',
        'CG': 'CG',
        'CZ': 'CZ',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD2': 'HD*',
        'HD3': 'HD*',
        'HE': 'HE',
        'HG2': 'HG*',
        'HG3': 'HG*',
        'HH11': 'HH*',
        'HH12': 'HH*',
        'HH21': 'HH*',
        'HH22': 'HH*',
        'NE': 'NE',
        'NH1': 'NH*',
        'NH2': 'NH*',
    },
    'TRP': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD1': 'CD1',
        'CD2': 'CD2',
        'CE2': 'CE2',
        'CE3': 'CE3',
        'CG': 'CG',
        'CH2': 'CH2',
        'CZ2': 'CZ2',
        'CZ3': 'CZ3',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD1': 'HD1',
        'HE1': 'HE1',
        'HE3': 'HE3',
        'HH2': 'HH2',
        'HZ2': 'HZ2',
        'HZ3': 'HZ3',
        'NE1': 'NE1',
    },
    'ALA': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB1': 'HB*',
        'HB2': 'HB*',
        'HB3': 'HB*',
    },
    'VAL': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CG1': 'CG*',
        'CG2': 'CG*',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB': 'HB',
        'HG11': 'HG*',
        'HG12': 'HG*',
        'HG13': 'HG*',
        'HG21': 'HG*',
        'HG22': 'HG*',
        'HG23': 'HG*',
    },
    'GLU': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD': 'CD',
        'CG': 'CG',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HE2': 'HE2',
        'HG2': 'HG*',
        'HG3': 'HG*',
        'OE1': 'OE1',
        'OE2': 'OE2',
    },
    'TYR': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CD1': 'CD*',
        'CD2': 'CD*',
        'CE1': 'CE*',
        'CE2': 'CE*',
        'CG': 'CG',
        'CZ': 'CZ',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HD1': 'HD*',
        'HD2': 'HD*',
        'HE1': 'HE*',
        'HE2': 'HE*',
        'HH': 'HH',
        'OH': 'OH',
    },
    'MET': {
        'C': 'C',
        'O': 'O',
        'OXT': 'O',
        'CA': 'CA',
        'N': 'N',
        'CB': 'CB',
        'CE': 'CE',
        'CG': 'CG',
        'H': 'H',
        'H1': 'H',
        'H2': 'H',
        'H3': 'H',
        'HA': 'HA',
        'HB2': 'HB*',
        'HB3': 'HB*',
        'HE1': 'HE*',
        'HE2': 'HE*',
        'HE3': 'HE*',
        'HG2': 'HG*',
        'HG3': 'HG*',
        'SD': 'SD',
    }
}



atom_ambiguity = defaultdict(
    lambda: defaultdict(lambda: 'Z'),
    [
        (k_0, defaultdict(
            lambda: 'Z',
            [
                (k_1, v_1)
                for k_1, v_1 in v_0.items()
            ]
        ))
        for k_0, v_0 in _atom_ambiguity.items()
    ]
)
adjacenty_matrices = dict()
for aa in bonds.keys():
    n_atoms = len(bonds[aa].keys())
    adjacenty_matrices[aa] = np.zeros((n_atoms, n_atoms))
    for atm1 in bonds[aa].keys():
        i = atom_names[aa].index(atm1)
        for atm2 in bonds[aa][atm1]:
            j = atom_names[aa].index(atm2)
            adjacenty_matrices[aa][i,j] = 1

distance_matrices = dict()
for aa in bonds.keys():
    n_atoms = len(bonds[aa].keys())
    distance_matrices[aa] = np.zeros((n_atoms, n_atoms))
    i = 1
    while np.any(distance_matrices[aa] == 0):
        idx = np.where(np.logical_and(
            distance_matrices[aa] == 0,
            np.linalg.matrix_power(adjacenty_matrices[aa],i) != 0
        ))
        distance_matrices[aa][idx] = i
        i += 1
    np.fill_diagonal(distance_matrices[aa], 0)

sparse_adjacenty_matrices = dict()
for aa in bonds.keys():
    n_atoms = len(bonds[aa].keys())
    sparse_adjacenty_matrices[aa] = ss.dok_matrix((n_atoms, n_atoms))
    for atm1 in bonds[aa].keys():
        i = atom_names[aa].index(atm1)
        for atm2 in bonds[aa][atm1]:
            j = atom_names[aa].index(atm2)
            sparse_adjacenty_matrices[aa][i,j] = 1

def atoms_within_bond_distance(res_type, atom_name, n):
    if res_type in one_to_three.keys():
        res_type = one_to_three[res_type]
    d = distance_matrices[res_type][atom_names[res_type].index(atom_name)]
    atoms = []
    for idx,e in enumerate(d):
        if isinstance(n, int) is int and e == n:
            atoms.append((res_type, atom_names[res_type][idx]))
            continue
        iter_n = iter(n)
        if e in n:
            atoms.append((res_type, atom_names[res_type][idx]))
            continue
        
    return atoms

# Tests to prove that the bonding is correct
def test_bonds(bonds):
    # All residues are closed
    for aa in bonds.keys():
        for atm1 in bonds[aa].keys():
            for atm2 in bonds[aa][atm1]:
                assert atm2 in bonds[aa].keys()

    # All bonds are symmetric
    for aa in bonds.keys():
        for atm1 in bonds[aa].keys():
            for atm2 in bonds[aa][atm1]:
                assert atm1 in bonds[aa][atm2]

    # All atoms have full valence:
    for aa in bonds.keys():
        for atm1 in bonds[aa].keys():
            if atm1[0] == 'H':
                assert len(bonds[aa][atm1]) == 1
            elif atm1[0] == 'O': # exception for double-bonded oxygens
                assert len(bonds[aa][atm1]) == 2 or (aa,atm1) in [('ASP','OD1'),('ASN','OD1'),('GLU','OE1'),('GLN','OE1'),]
            elif atm1[0] == 'S':
                assert len(bonds[aa][atm1]) == 2
            elif atm1[0] == 'N':
                assert len(bonds[aa][atm1]) == 3 or atm1 == 'N' or (aa == 'LYS' and atm1 == 'NZ')
            elif atm1[0] == 'C':
                assert len(bonds[aa][atm1]) != 4 or atm1 != 'C' or (aa,atm1) in [
                    # Methyls with indistinguishable protons
                    ('ALA','CB'),('ILE','CD1'),('ILE','CG2'),('LEU','CD1'),
                    ('LEU','CD2'),('MET','CE'),('THR','CG2'),('VAL','CG1'),
                    ('VAL', 'CG2'),
                    # Aromatic carbons
                    ('PHE','CG'),('PHE','CD1'),('PHE','CD2'),('PHE','CE1'),
                    ('PHE','CE2'),('PHE','CZ'),('TYR','CG'),('TYR','CD1'),
                    ('TYR','CD2'),('TYR','CE1'),('TYR','CE2'),('TYR','CZ'),
                    ('TRP','CD2'),('TRP','CE2'),('TRP','CE3'),('TRP','CZ2'),
                    ('TRP','CZ3'),('TRP','CH2'),('TRP','CG'),('TRP','CD1'),
                    ('HIS','CG'),('HIS','CD2'),('HIS','CE1'),
                    # Sidechain carboxyls
                    ('ASP','CG'),('GLU','CD'),
                    # Sidechain amides
                    ('ASN','CG'),('GLN','CD'),
                    # Arginine guanidinium
                    ('ARG','CZ')
                ]

#%%
def atom_to_index(sequence, res_num, atom_name):
    try:
        n = np.cumsum([len(atom_names[one_to_three[r]]) for r in sequence])
        return n[res_num-1] + atom_names[one_to_three[sequence[res_num]]].index(atom_name)
    except (IndexError, TypeError):
        return None


def index_to_atom(sequence, index):
    try:
        n = np.cumsum([len(atom_names[one_to_three[r]]) for r in sequence])
        idx = np.where(index < n)[0][0]
        r = sequence[idx]
        a_idx = index - (0 if idx == 0 else n[idx-1])
        return idx, atom_names[one_to_three[r]][a_idx]
    except (KeyError, IndexError, TypeError):
        return (None, None)


def sequence_to_adjacency_matrix(sequence):
    n = np.cumsum([len(atom_names[one_to_three[r]]) for r in sequence])
    n[0], n[1:] = 0, n[:-1]
    mat = ss.block_diag([
            sparse_adjacenty_matrices[one_to_three[r]]
            for r in sequence
        ],
        format='csr',
    )
    for i in range(len(sequence)-1):
        mat[n[i], n[i+1]+3] = -1
        mat[n[i+1]+3, n[i]] = -1
    return mat


#%%
