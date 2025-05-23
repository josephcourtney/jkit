from __future__ import annotations

from types import MethodType

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.cbook import Grouper
from matplotlib.colors import Normalize
from rich.console import Console
from rich.table import Table
from scipy.interpolate import interpn
from wskr.rich.plt import RichPlot, dpi_macbook_pro_13in_m2_2022, get_terminal_size

rng = np.random.default_rng()


def create_share_dict(share_param, ranges):
    """Create a dictionary to manage shared axes for subplots."""
    share_dic = {}
    if share_param:
        if share_param is True:
            for i in range(1, len(ranges)):
                share_dic[i] = 0
        elif isinstance(share_param, str):
            if share_param in {"row", "col"}:
                groups = {}
                for idx, (r0, _r1, c0, _c1) in enumerate(ranges):
                    key = r0 if share_param == "row" else c0
                    groups.setdefault(key, []).append(idx)
            else:
                msg = f"sharex must be bool, list of tuples, 'row', or 'col'; got {share_param!r}"
                raise ValueError(msg)
        elif (
            isinstance(share_param, list)
            and all(isinstance(e, tuple) for e in share_param)
            and all(isinstance(e_1, int) for e_0 in share_param for e_1 in e_0)
        ):
            for group in share_param:
                for subplot_index in group[1:]:
                    share_dic[subplot_index] = group[0]
            msg = f"sharex must be bool, list of tuples, 'row', or 'col'; got {share_param!r}"
            raise ValueError(msg)
    return share_dic


def check_for_overlaps(ranges, nrows, ncols):
    """Check for overlapping ranges in subplot configuration."""
    occupancy = np.zeros((nrows, ncols))
    for rng in ranges:
        occupancy[rng[0] : rng[1] + 1, rng[2] : rng[3] + 1] += 1
    if np.max(occupancy) > 1:
        msg = "Provided ranges cause overlapping subplots"
        raise ValueError(msg)
    return occupancy  # Return for potential future use


def initialize_subplots(
    fig, gs, ranges, sharex_dict, sharey_dict, sharez_dict, subplot_kwargs, *args, **kwargs
):
    """Initialize subplots within the figure."""
    ax = np.empty(len(ranges), dtype=object)
    for i, rng in enumerate(ranges):
        # merge base kwargs with any per-subplot overrides
        overrides = subplot_kwargs.get(i, {}) if subplot_kwargs else {}
        kwargs_ = {**kwargs, **overrides}

        share_x = ax[sharex_dict[i]] if i in sharex_dict else None
        share_y = ax[sharey_dict[i]] if i in sharey_dict else None
        if kwargs_.get("projection") == "3d" and i in sharez_dict:
            kwargs_["sharez"] = ax[sharez_dict[i]]

        axis = fig.add_subplot(
            gs[rng[0] : rng[1] + 1, rng[2] : rng[3] + 1],
            *args,
            sharex=share_x,
            sharey=share_y,
            **kwargs_,
        )

        # If this is a 3D subplot, give it get_shared_z_axes()
        if kwargs_.get("projection") == "3d":

            def get_shared_z_axes(self):
                return self._shared_z_axes

            axis.get_shared_z_axes = MethodType(get_shared_z_axes, axis)
        ax[i] = axis

    return ax


def make_plot_grid(
    nrows,
    ncols,
    *args,
    ranges=None,
    sharex=None,
    sharey=None,
    sharez=None,
    gs_kwargs=None,
    subplot_kwargs=None,
    figure_kwargs=None,
    return_gridspec=False,
    **kwargs,
):
    """Create a Matplotlib Figure and list of Axis objects with a given grid layout.

    Parameters
    ----------
    nrows, ncols: int, int
        Number of rows and columns for the subplot grid.

    ranges: list of tuples of ints (optional)
        A list of grid spaces for a given subplot to span. Grid spaces cannot
        be used more than once (no overlapping subplots) but do not have to be
        used at all. The index of each subplot in the returned list is given
        by its location in this list. By default, all grid spaces are separate
        subplots in left-right, top-down order.

    sharex: bool or list of tuples of ints [default: True]
        A list of groups of subplots that should share their x-axis, listed
        by the subplot's index. True causes all subplots to share their x-axis.
        By default, all x-axes are independent.

    sharey: bool or list of tuples of ints [default: True]
        A list of groups of subplots that should share their y-axis, listed
        by the subplot's index. True causes all subplots to share their y-axis.
        By default, all y-axes are independent.

    sharez: bool or list of tuples of ints (optional)
        A list of groups of subplots that should share their z-axis, listed
        by the subplot's index. This is applicable only for 3D subplots.

    gs_kwargs: dict
        Dictionary of keyword arguments to be passed to GridSpec when building
        subplot grid. Notable options are width_ratios and height_ratios.

    subplot_kwargs: dict
        Dictionary of keyword arguments to be passed to fig.add_subplot when
        initializing a new subplot. Note: any unrecognized arguments and
        keyword arguments passed to make_plot_grid are forwarded to
        fig.add_subplot.

    figure_kwargs: dict
        Dictionary of keyword arguments to be passed to plt.figure when
        initializing Figure object.

    return_gridspec: bool [default False]
        Whether to return the GridSpec object used in creation of the figure

    Returns
    -------
    fig: matplotlib.Figure
        Figure containing the subplots.

    ax: list of matplotlib.Axes
        List of Axis objects for each subplot, in the order specified in
        the ranges argument.

    gs: matplotlib.gridspec.GridSpec (optional)
        GridSpec instance used to create the subplots.

    """
    if sharez and sharez is not True:
        subplot_kwargs = subplot_kwargs or {}
        for group in sharez:
            for idx in group:
                subplot_kwargs.setdefault(idx, {}).setdefault("projection", "3d")

    gs_kwargs = gs_kwargs or {}
    figure_kwargs = figure_kwargs or {}

    if ranges is None:
        ranges = [(i, i, j, j) for i in range(nrows) for j in range(ncols)]

    check_for_overlaps(ranges, nrows, ncols)

    fig = plt.figure(**figure_kwargs)
    gs = fig.add_gridspec(nrows, ncols, **gs_kwargs)

    sharex_dict = create_share_dict(sharex, ranges)
    sharey_dict = create_share_dict(sharey, ranges)
    sharez_dict = create_share_dict(sharez, ranges)

    ax = initialize_subplots(
        fig,
        gs,
        ranges,
        sharex_dict,
        sharey_dict,
        sharez_dict,
        subplot_kwargs,
        *args,
        **kwargs,
    )

    # Handle 3D Z-axis sharing
    if sharez and sharez is not True:

        def _get_shared_z_axes(self):
            return self._shared_z_axes

        for group in sharez:
            g = Grouper()
            base_ax = ax[group[0]]
            for idx in group:
                g.join(base_ax, ax[idx])
                axis = ax[idx]
                axis._shared_z_axes = g  # noqa: SLF001
                axis.get_shared_z_axes = MethodType(_get_shared_z_axes, axis)

    return (fig, ax, gs) if return_gridspec else (fig, ax)


def stream_plot(t, arr, ax):
    """Generate a stream plot."""
    v = np.cumsum(arr, axis=1)
    ax.fill_between(t, 0, v[:, 0])
    for i in range(v.shape[1] - 1):
        ax.fill_between(t, v[:, i], v[:, i + 1])
    ax.fill_between(t, v[:, -1], np.sum(arr, axis=1))
    ax.set_xlim(0, np.max(t))
    ax.set_ylim(0, np.max(np.sum(arr, axis=1)))


def density_scatter(x, y, ax=None, bins=20, *, sort=True, **kwargs):
    """Scatter plot colored by 2d histogram."""
    if ax is None:
        _, ax = plt.subplots()

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
        fill_value=0,
    )

    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)
    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel("Density")

    return ax


def sparkline(x, y, columns=8, rows=1):
    fig, ax = make_plot_grid(1, 1)
    ax[0].plot(x, y, c="w")
    ax[0].axis("off")
    ax[0].margins(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return RichPlot(
        figure=fig,
        desired_width=columns,
        desired_height=rows,
        dpi=dpi_macbook_pro_13in_m2_2022,
    )


if __name__ == "__main__":
    console = Console()
    dpi = dpi_macbook_pro_13in_m2_2022
    w_px, h_px, n_col, n_row = get_terminal_size()
    print(f"cell size:       {w_px / n_col:>6.2f} x{h_px / n_row:>6.2f} pixels")
    print(f"cell resolution: {dpi * n_col / w_px:>6.2f} x{dpi * n_row / h_px:>6.2f} cell/inch")

    table = Table(title="Sparklines", show_lines=False)
    table.add_column("column 1", justify="left")
    table.add_column("column 2", justify="center")
    table.add_column("column 3", justify="right")
    for _i in range(3):
        table.add_row("name", "detail", sparkline(np.linspace(0, 1, 32), rng.normal(size=32)))
    console.print(table)

    fig, ax = make_plot_grid(1, 1, figure_kwargs={"layout": "constrained"})
    x = np.linspace(0, 1, 128)
    y = np.exp((2j * np.pi * 5 - 3) * x)
    ax[0].plot(x, y.real, c="w")
    ax[0].set_xlim(left=0)
    ax[0].spines["bottom"].set_position("zero")
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].xaxis.set_tick_params(which="both", direction="inout")
    ax[0].xaxis.set_tick_params(which="major", length=7.0)
    ax[0].xaxis.set_tick_params(which="minor", length=4.0)

    y_max = np.max([abs(e) for e in ax[0].get_ylim()])
    ax[0].set_ylim(-y_max, y_max)

    rp = RichPlot(figure=fig, desired_width=40, desired_height=20, dpi=dpi_macbook_pro_13in_m2_2022)
    console.print(rp)
