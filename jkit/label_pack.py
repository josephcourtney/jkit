#! /usr/bin/env python
# Create a contour plot of a 2D NMRPipe spectrum

import numpy as np
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import matplotlib.cm

from jkit.plot import make_plot_grid
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.patches as patches
from shapely.ops import cascaded_union
from shapely.affinity import translate
from shapely.geometry.polygon import orient
from tqdm import tqdm
import copy
from scipy.optimize import linear_sum_assignment


def get_base_contour(contours, extent):
    con = contours.collections[0]
    trans = con.get_transform()
    paths = con.get_paths()

    contour_list = []
    for con in contours.collections:
        for segNum, linepath in enumerate(con.get_paths()):
            contour_list.append(Polygon(linepath.vertices))

    base_contour = cascaded_union(contour_list)
    base_contour = base_contour.intersection(
        Polygon(
            [
                (extent[1], extent[2]),
                (extent[1], extent[3]),
                (extent[0], extent[3]),
                (extent[0], extent[2]),
            ]
        )
    )

    plt.close()

    return base_contour


def no_fit_polygon(base_pg, w, h):
    orbit_polygons = []
    for pg in base_pg:
        for pg_edge in [pg.exterior] + list(pg.interiors):
            coords = np.array(pg_edge.coords.xy).T
            edges = np.hstack((np.roll(coords, 1, 0), coords))
            for i in range(edges.shape[0]):
                orbit_polygons += [
                    Polygon(
                        [
                            (edges[i, 0] - w / 2, edges[i, 1] + h / 2),
                            (edges[i, 0] + w / 2, edges[i, 1] + h / 2),
                            (edges[i, 0] + w / 2, edges[i, 1] - h / 2),
                            (edges[i, 0] - w / 2, edges[i, 1] - h / 2),
                        ]
                    ),
                    Polygon(
                        [
                            (edges[i, 2] - w / 2, edges[i, 3] + h / 2),
                            (edges[i, 2] + w / 2, edges[i, 3] + h / 2),
                            (edges[i, 2] + w / 2, edges[i, 3] - h / 2),
                            (edges[i, 2] - w / 2, edges[i, 3] - h / 2),
                        ]
                    ),
                    Polygon(
                        [
                            (edges[i, 0] - w / 2, edges[i, 1] + h / 2),
                            (edges[i, 2] - w / 2, edges[i, 3] + h / 2),
                            (edges[i, 2] - w / 2, edges[i, 3] - h / 2),
                            (edges[i, 0] - w / 2, edges[i, 1] - h / 2),
                        ]
                    ),
                    Polygon(
                        [
                            (edges[i, 0] + w / 2, edges[i, 1] + h / 2),
                            (edges[i, 2] + w / 2, edges[i, 3] + h / 2),
                            (edges[i, 2] + w / 2, edges[i, 3] - h / 2),
                            (edges[i, 0] + w / 2, edges[i, 1] - h / 2),
                        ]
                    ),
                    Polygon(
                        [
                            (edges[i, 0] - w / 2, edges[i, 1] + h / 2),
                            (edges[i, 0] + w / 2, edges[i, 1] + h / 2),
                            (edges[i, 2] + w / 2, edges[i, 3] + h / 2),
                            (edges[i, 2] - w / 2, edges[i, 3] + h / 2),
                        ]
                    ),
                    Polygon(
                        [
                            (edges[i, 0] - w / 2, edges[i, 1] - h / 2),
                            (edges[i, 0] + w / 2, edges[i, 1] - h / 2),
                            (edges[i, 2] + w / 2, edges[i, 3] - h / 2),
                            (edges[i, 2] - w / 2, edges[i, 3] - h / 2),
                        ]
                    ),
                ]
    orbit_polygons = [orient(pg) for pg in orbit_polygons if pg.area > 0]
    orbit_union = cascaded_union([base_pg] + orbit_polygons)
    orbit_union = orbit_union
    return orbit_union


def place_boxes(
    nfp, box_nfp, extent, box_dim, progress=True, max_boxes=np.inf, angle_cutoff=270
):
    extent = [min(extent[:2]), max(extent[:2]), min(extent[2:]), max(extent[2:])]
    boxes = []
    max_verts = 1
    with tqdm(total=max_verts, unit="vertices") as bar:
        if not progress:
            bar.disable()
        while len(boxes) < max_boxes:
            if isinstance(nfp, MultiPolygon):
                verts = [
                    np.array(pg_edge.coords.xy).T
                    for pg in nfp
                    for pg_edge in ([pg.exterior] + list(pg.interiors))
                ]
            else:
                verts = [
                    np.array(pg_edge.coords.xy).T
                    for pg_edge in ([nfp.exterior] + list(nfp.interiors))
                ]
            verts = [
                np.array(
                    [
                        v
                        for v in verts_sub
                        if (
                            v[0] > extent[0] + box_dim[0] / 2
                            and v[0] < extent[1] - box_dim[0] / 2
                            and v[1] > extent[2] + box_dim[1] / 2
                            and v[1] < extent[3] - box_dim[1] / 2
                        )
                    ]
                )
                for verts_sub in verts
                if verts_sub.ndim == 2 and verts_sub.shape[0] > 0
            ]

            angles = [
                [
                    np.fmod(
                        2 * np.pi
                        + (
                            np.arctan2(
                                verts[i][(j + 1) % len(verts[i])][1]
                                - verts[i][(j) % len(verts[i])][1],
                                verts[i][(j + 1) % len(verts[i])][0]
                                - verts[i][(j) % len(verts[i])][0],
                            )
                            - np.arctan2(
                                verts[i][(j - 1) % len(verts[i])][1]
                                - verts[i][(j) % len(verts[i])][1],
                                verts[i][(j - 1) % len(verts[i])][0]
                                - verts[i][(j) % len(verts[i])][0],
                            )
                        ),
                        2 * np.pi,
                    )
                    for j in range(len(verts[i]))
                ]
                for i in range(len(verts))
            ]

            verts = [
                verts[i][j]
                for i in range(len(verts))
                for j in range(len(verts[i]))
                if angles[i][j] >= angle_cutoff * (np.pi / 180)
            ]

            if len(verts) == 0:
                break
            tot_verts = np.vstack(verts)
            if tot_verts.shape[0] > max_verts:
                max_verts = tot_verts.shape[0]
                bar.total = max_verts
            if tot_verts.shape[0] < 2:
                break

            bar.n = max_verts - tot_verts.shape[0]
            bar.refresh()

            i = np.random.randint(len(verts))
            new_box = verts[i]
            boxes.append(new_box)
            new_box_nfp = translate(box_nfp, new_box[0], new_box[1])
            nfp = cascaded_union([nfp, new_box_nfp])

    return nfp, boxes


def plot_progress(base_contour, nfp_orig, nfp_final, boxes, extent):
    fig, ax = make_plot_grid(1, 1)
    for pg in base_contour:
        for pg_edge in [pg.exterior] + list(pg.interiors):
            pg_patch = patches.Polygon(
                np.array(pg_edge.coords.xy).T,
                facecolor="none",
                edgecolor="g",
                alpha=0.5,
            )
            ax[0].add_patch(pg_patch)

    if not isinstance(nfp_orig, MultiPolygon):
        nfp_orig = [nfp_orig]
    for pg in nfp_orig:
        for pg_edge in [pg.exterior] + list(pg.interiors):
            pg_patch = patches.Polygon(
                np.array(pg_edge.coords.xy).T,
                edgecolor="k",
                facecolor="none",
                alpha=0.5,
            )
            ax[0].add_patch(pg_patch)

    if not isinstance(nfp_final, MultiPolygon):
        nfp_final = [nfp_final]
    for pg in nfp_final:
        for pg_edge in [pg.exterior] + list(pg.interiors):
            pg_patch = patches.Polygon(
                np.array(pg_edge.coords.xy).T,
                edgecolor="k",
                facecolor="none",
                alpha=0.5,
            )
            ax[0].add_patch(pg_patch)

    for v in boxes[:-1]:
        coords = np.array(
            [
                (v[0] - label_dim[0] / 2, v[1] - label_dim[1] / 2),
                (v[0] + label_dim[0] / 2, v[1] - label_dim[1] / 2),
                (v[0] + label_dim[0] / 2, v[1] + label_dim[1] / 2),
                (v[0] - label_dim[0] / 2, v[1] + label_dim[1] / 2),
            ]
        )
        pg_patch = patches.Polygon(coords, facecolor="r", edgecolor="k", alpha=0.5,)
        ax[0].add_patch(pg_patch)

    coords = np.array(
        [
            (boxes[-1][0] - label_dim[0] / 2, boxes[-1][1] - label_dim[1] / 2),
            (boxes[-1][0] + label_dim[0] / 2, boxes[-1][1] - label_dim[1] / 2),
            (boxes[-1][0] + label_dim[0] / 2, boxes[-1][1] + label_dim[1] / 2),
            (boxes[-1][0] - label_dim[0] / 2, boxes[-1][1] + label_dim[1] / 2),
        ]
    )
    pg_patch = patches.Polygon(coords, facecolor="k", edgecolor="r", alpha=0.5,)
    ax[0].add_patch(pg_patch)

    ax[0].set_xlim(*extent[:2])
    ax[0].set_ylim(*extent[2:])
    ax[0].invert_xaxis()
    ax[0].invert_yaxis()


def pack_labels(base_contour, extent, label_dim, markers):
    # find closest position allowed for labels to contours
    nfp = no_fit_polygon(base_contour, label_dim[0], label_dim[1])
    box_nfp = no_fit_polygon(
        MultiPolygon(
            [
                Polygon(
                    [
                        (-label_dim[0] / 2, -label_dim[1] / 2),
                        (label_dim[0] / 2, -label_dim[1] / 2),
                        (label_dim[0] / 2, label_dim[1] / 2),
                        (-label_dim[0] / 2, label_dim[1] / 2),
                    ]
                )
            ]
        ),
        label_dim[0],
        label_dim[1],
    )
    # pack the maximum possible number of label boxes in available space
    nfp_final, boxes = place_boxes(nfp, box_nfp, extent, label_dim)

    _boxes = np.copy(boxes)
    _markers = np.copy(markers)
    _boxes[:, 1] /= 10
    _markers[:, 1] /= 10

    cost = np.sqrt(np.sum((_markers[:, None, :] - _boxes[None, :, :]) ** 2, 2))

    # select optimal matching of marker and label positions
    marker_idx, label_idx = linear_sum_assignment(cost)

    return marker_idx, label_idx, np.array(boxes)


if __name__ == "__main__":
    import nmrglue as ng
    from jkit import nmr

    path = "/u/jcourtney/Sparky/spackages/ubq_pjump_noesy/Save/Ubq_pjump_folded_298K_NOESy_HSQC_1812_800MHz_scale_hn_15n.save"
    print("reading sparky save file")
    spec = nmr.SparkySpectrum(
        path,
        temperature=0,
        pressure=1,
        pH=7.0,
        concentration=1.0,
        sample_id=0,
        offset=(None),
    )

    print("reading sparky data file")
    sdic, sdata = ng.sparky.read(spec.ucsf)

    # convert to NMRPipe format
    C = ng.convert.converter()
    C.from_sparky(sdic, sdata)
    dic, data = C.to_pipe()

    # make ppm scales
    uc_1h = ng.pipe.make_uc(dic, data, dim=1)
    ppm_1h = uc_1h.ppm_scale()
    ppm_1h_0, ppm_1h_1 = uc_1h.ppm_limits()
    uc_15n = ng.pipe.make_uc(dic, data, dim=0)
    ppm_15n = uc_15n.ppm_scale()
    ppm_15n_0, ppm_15n_1 = uc_15n.ppm_limits()
    xlim = (7, 10)
    ylim = (100, 135)
    label_dim = (0.13, 0.45)

    # plot parameters
    cmap = matplotlib.cm.Blues_r  # contour map (colors to use for contours)
    contour_start = 32.5  # contour level start value
    contour_num = 10  # number of contour levels
    contour_factor = 2.0  # scaling factor between contour levels

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)

    contours = plt.contour(data, levels=[contour_start], extent=extent)
    base_contour = get_base_contour(contours)

    marker_idx, label_idx, boxes = pack_labels(
        spec=spec,
        data=data,
        extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1),
        label_dim=label_dim,
        markers=spec.peak_dataframe()[["w_2", "w_1"]].values,
    )

    # create the figure
    fig = plt.figure(figsize=(8.0, 10.5))
    ax = fig.add_subplot(111)

    # plot the contours
    contours = ax.contour(
        data,
        cl,
        cmap=cmap,
        extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1),
        linewidths=0.5,
    )

    labels = []

    for i, pk in spec.peak_dataframe().iterrows():
        atom = pk.res_type_2
        state_1 = pk.atom_1[-2].upper()
        state_2 = pk.atom_2[-2].upper()

        d = np.sqrt(
            (boxes[label_idx[i], 0] - pk.w_2) ** 2
            + ((boxes[label_idx[i], 1] - pk.w_1) / 10) ** 2
        )

        ax.scatter(
            pk.w_2, pk.w_1, marker="x", linewidths=0.02, s=10, c="k",
        )

        if d > 0.01:
            ax.annotate(
                s=f"${atom}^{{{state_1}{state_2}}}$",
                xy=(pk.w_2, pk.w_1),
                xytext=(boxes[label_idx[i], 0], boxes[label_idx[i], 1]),
                size=6,
                color="k",
                horizontalalignment="center",
                verticalalignment="center",
                arrowprops=dict(arrowstyle="-", lw=1, color="k", shrinkA=0, shrinkB=0,),
            )
        else:
            ax.text(
                x=boxes[label_idx[i], 0],
                y=boxes[label_idx[i], 1],
                s=f"${atom}^{{{state_1}{state_2}}}$",
                fontsize=6,
                color="k",
                horizontalalignment="center",
                verticalalignment="center",
            )

    # decorate the axes
    ax.set_ylabel(r"$F_{1}$ $(^{15}N,$ $ppm)$")
    ax.set_xlabel(r"$F_{2}$ $(^{1}H,$ $ppm)$")
    ax.set_title("reverse HSQC")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlim(9.6, 7)
    # ax.set_ylim(181.6, 171.6)
    ax.set_aspect(0.1)

    # fig.savefig("LPHP_HSQC_reverse.pdf")
    plt.show()
