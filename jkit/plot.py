import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interpn

# plt.rc(
#     "axes",
#     prop_cycle=(
#         cycler("color", ["#c44a2c", "#de8a2e", "#edda71", "#c7bb81", "#34a69f"])
#     ),
# )


def make_plot_grid(
    nrows,
    ncols,
    ranges=None,
    sharex=None,
    sharey=None,
    sharez=None,
    gs_kwargs={},
    subplot_kwargs=None,
    figure_kwargs=None,
    *args,
    **kwargs
):
    """Create a Matplotlib Figure and list of Axis objects with a given grid
    layout.

    Parameters
    ----------
    nrows, ncols: int, int
        number of rows and columns for the subplot grid

    ranges: list of tuples of ints (optional)
        A list of grid spaces for a given subplot to span. Grid spaces cannot
        be used more than once (no overlapping subplots) but do not have to be
        used at all. The index of each subplot in the returned list is given
        by its location in this list. By default, all grid spaces are separate
        subplots in left-right, top-down order.

    sharex: True or list of tuples of ints
        A list of groups of subplots that should share their x-axis, listed
        by the subplot's index. True causes all subplots to share their x-axis.
        By default, all x-axes are independent.

    sharey: True or list of tuples of ints
        A list of groups of subplots that should share their y-axis, listed
        by the subplot's index. True causes all subplots to share their y-axis.
        By default, all y-axes are independent.

    gs_kwargs: dictionary of keyword arguments
        Dictionary of keyword arguments to be passed to GridSpec when building
        subplot grid. Notable options are width_ratios and height_ratios


    subplot_kwargs: dictionary of keyword arguments
        Dictionary of keyword arguments to be passed to fig.add_subplot when
        initializing a new subplot. Note: any unrecognized arguments and
        keyword arguments passed to make_plot_grid are forwarded to
        fig.add_subplot.

    figure_kwargs: dictionary of keyword arguments
        Dictionary of keyword arguments to be passed to plt.figure when
        initializing Figure object.

    Returns
    -------
    fig: matplotlib.Figure object
        Figure containing the subplots.

    ax: list of matplotlib.Axis objects
        List of Axis objects for each subplot, in the order specified in
        the ranges argument.
    """

    occupancy = np.zeros((nrows, ncols))
    if ranges is None:
        ranges = [(i, i, j, j) for i in range(nrows) for j in range(ncols)]
    for rng in ranges:
        occupancy[rng[0] : rng[1] + 1, rng[2] : rng[3] + 1] += 1
    assert np.max(occupancy) < 2
    ax = np.empty(len(ranges), dtype=object)

    sharex_dic = dict()
    if sharex is not None:
        if sharex is True:
            for i in range(1, nrows * ncols):
                sharex_dic[i] = 0
        else:
            for grp in sharex:
                for e in list(grp)[1:]:
                    sharex_dic[e] = list(grp)[0]

    sharey_dic = dict()
    if sharey is not None:
        if sharey is True:
            for i in range(1, nrows * ncols):
                sharey_dic[i] = 0
        else:
            for grp in sharey:
                for e in list(grp)[1:]:
                    sharey_dic[e] = list(grp)[0]

    sharez_dic = dict()
    if sharez is not None:
        if sharez is True:
            for i in range(1, nrows * ncols):
                sharez_dic[i] = 0
        else:
            for grp in sharez:
                for e in list(grp)[1:]:
                    sharez_dic[e] = list(grp)[0]

    _figure_kwargs = {}
    if figure_kwargs is not None:
        _figure_kwargs.update(figure_kwargs)
    fig = plt.figure(**_figure_kwargs)
    gs = fig.add_gridspec(nrows, ncols, **gs_kwargs)
    for i, rng in enumerate(ranges):
        if subplot_kwargs is not None and i in subplot_kwargs.keys():
            _kwargs = {k: v for k, v in kwargs.items()}
            _kwargs.update(subplot_kwargs[i])
        else:
            _kwargs = kwargs

        if i in sharex_dic.keys():
            shrx = ax[sharex_dic[i]]
        else:
            shrx = None

        if i in sharey_dic.keys():
            shry = ax[sharey_dic[i]]
        else:
            shry = None

        if _kwargs.get("projection", None) == "3d":
            if i in sharez_dic.keys():
                _kwargs["sharez"] = ax[sharez_dic[i]]
            else:
                _kwargs["sharez"] = None
                shrz = None

        ax[i] = fig.add_subplot(
            gs[rng[0] : rng[1] + 1, rng[2] : rng[3] + 1], sharex=shrx, sharey=shry, *args, **_kwargs
        )

    return fig, ax


def stream_plot(t, arr, ax):
    v = np.cumsum(arr, 1)
    ax.fill_between(t, 0, v[:, 0])
    for i in range(v.shape[1] - 1):
        ax.fill_between(t, v[:, i], v[:, i + 1])
    ax.fill_between(t, v[:, -1], np.sum(arr, -1))
    ax.set_xlim(0, np.max(t))
    ax.set_ylim(0, np.max(np.sum(arr, -1)))


def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None:
        fig, ax = plt.subplots()

    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel("Density")

    return ax
