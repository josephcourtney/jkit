#!/usr/bin/env python
# coding:utf-8

import collections
import numpy as np
import pandas as pd
import re
from pprint import PrettyPrinter

pprint = PrettyPrinter(indent=4).pprint
import nmrglue as ng
from . import nmr, pdb, aminoacids
import scipy.special
from collections.abc import Iterable


def stringlike(e):
    return isinstance(e, (str, bytes, bytearray))


# NCACX Pathways
ncacx_primary = [  # Standard pathway with moderate mixing
    (  # A atom specification
        None,  # restrictions of sequence position relative to first atom
        None,  # residue type restrictions
        ["N"],  # atom type restrictions
        None,  # restrictions on number of bonds or distance from previous atom
    ),
    ([0], None, ["CA"], None),
    (
        [-2, -1, 0, 1, 2],  # multiple options are specified with list
        None,
        ["C.*"],  # atom type restrictions are specified by regexp
        None,  # additional restrictions on peak types
    ),
]
ncacx_lysine = [
    (None, ["K"], ["NZ"], None),
    ([0], ["K"], ["CE"], None),
    ([-2, -1, 0, 1, 2], None, ["C.*"], None),
]  # Lysine pathway
ncacx_arginine = [
    (None, ["R"], ["NE"], None),
    ([0], ["R"], ["CD"], None),
    ([-2, -1, 0, 1, 2], None, ["C.*"], None),
]  # Arginine pathway
ncacx_proline = [
    (None, ["P"], ["N"], None),
    ([0], ["P"], ["CD"], None),
    ([-2, -1, 0, 1, 2], None, ["C.*"], None),
]  # Proline NCDCX pathway

# NCACBCO Pathway
ncacbco_primary = [
    (None, None, ["N"], None),
    ([0], None, ["CA"], None),
    (
        [-2, -1, 0, 1, 2],
        None,
        ["C.*"],
        ("b", [0, 1]),
    ),  # peaks are restricted to 0 or 1 bonds away
]

# CANCO
canco_primary = [
    (None, None, ["CA"], None),
    ([0], None, ["N"], None),
    ([-1], None, ["C"], None),
]

# NCOCX Pathways
ncocx_primary = [
    (None, None, ["N"], None),
    ([-1], None, ["C"], None),
    ([0], None, ["C.*"], None),
]
ncocx_asparagine = [
    (None, ["N"], ["ND2"], None),
    ([0], ["N"], ["CG"], None),
    ([0], ["N"], ["C.*"], None),
]  # Asparagine sidechain pathway
ncocx_glutamine = [
    (None, ["Q"], ["NE2"], None),
    ([0], ["Q"], ["CD"], None),
    ([0], ["Q"], ["C.*"], None),
]  # Glutamine pathway



# long-mixing NCOCX Pathways
ncocx_long_primary = [
    (None, None, ["N"], None),
    ([-1], None, ["C"], None),
    (None, None, ["C.*"], ("s", [5.0])),  # atoms must be within 5 angstroms
]
ncocx_long_asparagine = [
    (None, ["N"], ["ND2"], None),
    ([0], ["N"], ["CG"], None),
    (None, None, ["C.*"], ("s", [5.0])),  # atoms must be within 5 angstroms
]  # Asparagine sidechain pathway
ncocx_long_glutamine = [
    (None, ["Q"], ["NE2"], None),
    ([0], ["Q"], ["CD"], None),
    (None, None, ["C.*"], ("s", [5.0])),  # atoms must be within 5 angstroms
]  # Glutamine pathway


# NN Pathways
long_nn = [
    (None, None, ["N.*"], None),
    (None, None, ["N.*"], ("s", [5.0])),  # atoms must be within 5 angstroms
]


# CC Pathways
long_cc = [
    (None, None, ["C.*"], None),
    (None, None, ["C.*"], ("s", [8.5])),  # atoms must be within 5 angstroms
]

medium_cc = [(None, None, ["C.*"], None), ([-2, -1, 0, 1, 2], None, ["C.*"], None)]

short_cc = [(None, None, ["C.*"], None), ([0], None, ["C.*"], None)]

# HSQC Pathways
hsqc_primary = [(None, None, ["H.*"], None), ([0], None, ["N.*"], ("b", [1]))]

# HNCO
hnco_primary = [
    (None, None, ["H"], None),
    ([0], None, ["N"], None),
    ([-1], None, ["C"], None),
]

# NOESy-HSQC HNH
noesy_hsqc_hnh_long_range = [
    (None, None, ["H.*"], None),
    (None, None, ["H.*"], ("s", [5.0])),
    ([0], None, ["N.*"], ("b", [1])),
]
noesy_hsqc_hnh_diagonal = [
    (None, None, ["H.*"], None),
    (None, None, ["H.*"], ("b", [0, 1, 2, 3, 4, 5, 6])),
    ([0], None, ["N.*"], ("b", [1])),
]

pathway_dict = dict(
    ncacx_primary=[ncacx_primary],
    ncacx_lysine=[ncacx_lysine],
    ncacx_arginine=[ncacx_arginine],
    ncacx_proline=[ncacx_proline],
    ncacbco_primary=[ncacbco_primary],
    canco_primary=[canco_primary],
    ncocx_primary=[ncocx_primary],
    ncocx_asparagine=[ncocx_asparagine],
    ncocx_glutamine=[ncocx_glutamine],
    ncocx_long_primary=[ncocx_long_primary],
    ncocx_long_asparagine=[ncocx_long_asparagine],
    ncocx_long_glutamine=[ncocx_long_glutamine],
    long_nn=[long_nn],
    long_cc=[long_cc],
    medium_cc=[medium_cc],
    short_cc=[short_cc],
    hsqc=[hsqc_primary],
    hnco_primary=[hnco_primary],
    hnco=[hnco_primary],
    ncocx=[ncocx_primary, ncocx_asparagine, ncocx_glutamine],
    ncocx_long=[ncocx_long_primary, ncocx_long_asparagine, ncocx_long_glutamine],
    ncacx=[ncacbco_primary, ncacx_arginine, ncacx_lysine, ncacx_proline],
    noesy_hsqc_hnh=[noesy_hsqc_hnh_diagonal, noesy_hsqc_hnh_long_range],
)


def Tree():
    return collections.defaultdict(Tree)


def walk_tree(t, depth=0, max_depth=3, path=[]):
    if depth == max_depth:
        yield path
    else:
        for k in t.keys():
            depth += 1
            for x in walk_tree(t[k], depth, max_depth, path + [k]):
                yield x
            depth -= 1


def enumerate_correlations(
    pathway,
    path_tree=Tree(),
    depth=0,
    previous_atom=None,
    atom_sets=None,
    transfer_sets=None,
):
    if len(pathway) == depth:
        return path_tree

    for atom in atom_sets[depth]:
        if (
            depth == 0
            or (
                transfer_sets[depth - 1] is None
                or (previous_atom, atom) in transfer_sets[depth - 1]
            )
            and (
                pathway[depth][0] is None
                or (atom[1] - previous_atom[1]) in pathway[depth][0]
            )
        ):
            path_tree[atom] = enumerate_correlations(
                pathway=pathway,
                path_tree=Tree(),
                depth=depth + 1,
                previous_atom=atom,
                atom_sets=atom_sets,
                transfer_sets=transfer_sets,
            )
    return path_tree


def correlations_matching_pathways(structure, pathways, models=None):
    sequence = structure.sequence[0]
    if models is None:
        models = [0]
    if isinstance(models, Iterable) and not stringlike(models):
        models = list(models)
    else:
        models = [models]

    if stringlike(pathways):
        pathways = pathway_dict[pathways]
    elif isinstance(pathways, Iterable) and not stringlike(pathways):
        _pathways = []
        for pw in pathways:
            if stringlike(pw):
                pathway_name = bytearray().extend(pw).decode("utf-8")
                _pathways.append(pathway_dict[pathway_name])
            elif isinstance(pw, Iterable):
                if all(
                    [
                        isinstance(e, Iterable) and not stringlike(e) and len(e) == 4
                        for e in pw
                    ]
                ):
                    _pathways.append(pw)
        pathways = _pathways
    nd = len(pathways[0])
    assert all([len(p) == nd for p in pathways])

    df = None

    for pathway in pathways:
        # apply atom type restrictions
        atom_restrictions = [None] * (len(pathway))
        for i in range(len(pathway)):
            if not pathway[i][2] is None:
                atom_restrictions[i] = []
                for res_num in range(1, len(sequence) + 1):
                    # apply residue type restriction
                    if pathway[i][1] is None or sequence[res_num - 1] in pathway[i][1]:
                        for atom_name in aminoacids.atom_names[
                            aminoacids.one_to_three[sequence[res_num - 1]]
                        ]:
                            # apply atom name restriction
                            if any(
                                [
                                    re.match(re.compile("^" + pattern + "$"), atom_name)
                                    for pattern in pathway[i][2]
                                ]
                            ):
                                atom_restrictions[i].append(
                                    (sequence[res_num - 1], res_num, atom_name)
                                )
        # apply transfer restrictions
        transfer_restrictions = [None] * (len(pathway) - 1)
        for i in range(1, len(pathway)):
            # find the set of all transfers that abide by restrictions on spatial distance
            if pathway[i][3] is None:
                continue

            transfer_restrictions[i - 1] = []
            allowed_pairs = []
            disallowed_pairs = []
            if pathway[i][3][0] == "s":
                for mdl_num in models:
                    pdb_model = structure.models[mdl_num]
                    allowed_pairs += [
                        (
                            (
                                aminoacids.three_to_one[atm1.res_name],
                                atm1.res_num,
                                atm1.atom_name,
                            ),
                            (
                                aminoacids.three_to_one[atm2.res_name],
                                atm2.res_num,
                                atm2.atom_name,
                            ),
                        )
                        for atm1, atm2 in pdb_model.all_atom_pairs_within_distance(
                            pathway[i][3][1][-1]
                        )
                    ]

                    if len(pathway[i][3][1]) == 2:
                        disallowed_pairs += [
                            (
                                (
                                    aminoacids.three_to_one[atm1.res_name],
                                    atm1.res_num,
                                    atm1.atom_name,
                                ),
                                (
                                    aminoacids.three_to_one[atm2.res_name],
                                    atm2.res_num,
                                    atm2.atom_name,
                                ),
                            )
                            for atm1, atm2 in pdb_model.all_atom_pairs_within_distance(
                                pathway[i][3][1][0]
                            )
                        ]
            elif pathway[i][3][0] == "b":
                for atm1 in atom_restrictions[i - 1]:
                    for a in aminoacids.atoms_within_bond_distance(
                        atm1[0], atm1[2], pathway[i][3][1]
                    ):
                        atm2 = (aminoacids.three_to_one[a[0]], atm1[1], a[1])
                        if atm2 in atom_restrictions[i]:
                            allowed_pairs.append((atm1, atm2))

            for pair in allowed_pairs:
                if (
                    not pair in disallowed_pairs
                    and pair[0] in atom_restrictions[i - 1]
                    and pair[1] in atom_restrictions[i]
                ):
                    transfer_restrictions[i - 1].append(pair)

        peak_tree = enumerate_correlations(
            pathway=pathway,
            atom_sets=atom_restrictions,
            transfer_sets=transfer_restrictions,
        )

        tmp_df = pd.DataFrame(
            data=[
                tuple(sum([list(e2) for e2 in e1], []))
                for e1 in walk_tree(peak_tree, max_depth=len(pathway))
            ],
            columns=sum(
                [
                    [
                        "residue_type_{}".format(i),
                        "residue_number_{}".format(i),
                        "atom_name_{}".format(i),
                    ]
                    for i in range(nd)
                ],
                [],
            ),
        )
        tmp_df = tmp_df.drop_duplicates()
        if df is None:
            df = tmp_df
        else:
            df = df.append(tmp_df)

    return df


def simulate_peaks(structure, pathways, chemical_shifts, models=None):
    peaks = correlations_matching_pathways(structure, pathways, models)
    nd = len(peaks.columns) // 3
    for i in range(nd):
        peaks = pd.merge(
            peaks,
            chemical_shifts,
            how="left",
            left_on=[
                "residue_number_{}".format(i),
                "residue_type_{}".format(i),
                "atom_name_{}".format(i),
            ],
            right_on=["residue_number", "residue_type", "atom"],
        )
        peaks.rename(
            columns={"w": "w_{}".format(i), "w_err": "w_{}_err".format(i)}, inplace=True
        )
        peaks.drop(["residue_number", "residue_type", "atom"], axis=1, inplace=True)
    return peaks



def simulate_spectrum(
    peaks,
    udic,
    dim_order=None,
    lw={"H": 10, "N": 10, "C": 10},
    amplitude=1.0,
    sigma=0.0,
    lineshape=None,
    format="sparky",
    write=True,
    fname="spec",
):
    peaks = peaks.dropna()
    ndim = udic["ndim"]

    if format == "sparky":
        dic = ng.sparky.create_dic(udic)

        # convert the peak list from PPM to points
        uc_objects = [ng.sparky.make_uc(dic, None, i) for i in range(ndim)]
    elif format == "nmrpipe":
        dic = ng.pipe.create_dic(udic)
        shape = [udic[i]["size"] for i in range(ndim)]
        _data = np.empty(shape)

        # convert the peak list from PPM to points
        uc_objects = [ng.pipe.make_uc(dic, _data, i) for i in range(ndim)]

    if lineshape is None:
        lineshape = tuple("l" for _ in range(ndim))

    if dim_order is None:
        dim_order = tuple(range(ndim))

    lw_pts = [
        abs(
            uc_objects[i].f(lw[udic[dim_order[i]]["label"][-1]], "Hz")
            - uc_objects[i].f(0, "Hz")
        )
        for i in range(ndim)
    ]

    params = []
    amplitudes = []
    for _, pk in peaks.iterrows():
        pk_param = []
        for i in range(ndim):
            idx = dim_order[i]
            pk_param.append(
                (uc_objects[i].f(pk[f"w_{idx}"], "ppm"), pk.get(f"lw_{idx}", lw_pts[i]))
            )
        if not np.any(np.isnan(pk_param)):
            params.append(pk_param)
            amplitudes.append(pk.get("height", amplitude))

    # simulate the spectrum
    shape = tuple(udic[i]["size"] for i in range(ndim))

    data = ng.linesh.sim_NDregion(shape, lineshape, params, amplitudes)

    data += sigma * np.random.normal(size=shape)

    if write:
        if format == "sparky":
            ng.sparky.write(
                f"{fname}.ucsf", dic, data.astype("float32"), overwrite=True
            )
        elif format == "nmrpipe":
            ng.pipe.write(
                f"{fname}.ft{ndim}", dic, data.astype("float32"), overwrite=True
            )

    return dic, data



def remove_overlapping_peaks(pk_set_1, pk_set_2, r):
    # remove all peaks in pk_set_1 that overlap with peaks in pk_set_2
    ws = [c for c in list(pk_set_1.columns) if c[:2] == "w_"]
    tree_1 = scipy.spatial.KDTree(pk_set_1.as_matrix(ws))
    tree_2 = scipy.spatial.KDTree(pk_set_2.as_matrix(ws))
    overlapped_pts = list(set(sum(tree_2.query_ball_tree(tree_1, r=r), [])))
    pk_set_1 = pk_set_1.reset_index()
    return pk_set_1.drop(pk_set_1.index[overlapped_pts])


def sparsify(df, r):
    def simplex_volume(pts):
        return np.linalg.det(pts[1:, :] - points[0, :]) / np.factorial(pts.shape[0])

    def get_point_volumes(voro):
        volumes = np.zeros(voro.ridge_vertices.shape[0])
        for i in range(voro.ridge_vertices.shape[0]):
            volumes[i] = simplex_volume(voro.vertices[voro.ridge_vertices[i, :], :])
        return volumes

    # remove peaks in high density areas one-by-one according to peak distance until minimum distances <= r
    ws = [c for c in list(pk_set_1.columns) if c[:2] == "w_"]
    pts = df.as_matrix(ws)
    voro = scipy.spatial.Voronoi(pts)
    volumes = get_point_volumes(voro)

    tree_2 = scipy.spatial.KDTree(pk_set_2.as_matrix(ws))
    overlapped_pts = list(set(sum(tree_2.query_ball_tree(tree_1, r=r), [])))
    pk_set_1 = pk_set_1.reset_index()
    return pk_set_1.drop(pk_set_1.index[overlapped_pts])
