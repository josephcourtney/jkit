import matplotlib.pyplot as plt
import numpy as np
import pytest

from jkit.plot import make_plot_grid

rng = np.random.default_rng()


def test_basic_grid():
    fig, ax = make_plot_grid(2, 2)
    assert len(ax) == 4
    assert isinstance(fig, plt.Figure)
    for axis in ax:
        assert isinstance(axis, plt.Axes)


def test_custom_ranges():
    ranges = [(0, 0, 0, 1), (1, 1, 0, 1)]
    fig, ax = make_plot_grid(2, 2, ranges=ranges)
    assert len(ax) == 2
    assert isinstance(fig, plt.Figure)
    for axis in ax:
        assert isinstance(axis, plt.Axes)


def test_sharex_true():
    _fig, ax = make_plot_grid(2, 2, sharex=True)
    assert len(ax) == 4
    for axis in ax[1:]:
        assert axis.get_shared_x_axes().joined(ax[0], axis)


def test_sharey_true():
    _fig, ax = make_plot_grid(2, 2, sharey=True)
    assert len(ax) == 4
    for axis in ax[1:]:
        assert axis.get_shared_y_axes().joined(ax[0], axis)


def test_sharex_specific():
    sharex = [(0, 1)]
    _fig, ax = make_plot_grid(2, 2, sharex=sharex)
    assert len(ax) == 4
    assert ax[0].get_shared_x_axes().joined(ax[0], ax[1])
    assert not ax[0].get_shared_x_axes().joined(ax[0], ax[2])


def test_sharey_specific():
    sharey = [(0, 2)]
    _fig, ax = make_plot_grid(2, 2, sharey=sharey)
    assert len(ax) == 4
    assert ax[0].get_shared_y_axes().joined(ax[0], ax[2])
    assert not ax[0].get_shared_y_axes().joined(ax[0], ax[1])


def test_no_overlap():
    ranges = [(0, 0, 0, 1), (1, 1, 0, 1)]
    _fig, ax = make_plot_grid(2, 2, ranges=ranges)
    assert len(ax) == 2


def test_overlap_raises_error():
    ranges = [(0, 0, 0, 1), (0, 1, 0, 1)]
    with pytest.raises(ValueError, match="Provided ranges cause overlapping subplots"):
        make_plot_grid(2, 2, ranges=ranges)


def test_subplot_kwargs():
    subplot_kwargs = {0: {"title": "First Plot"}}
    _fig, ax = make_plot_grid(2, 2, subplot_kwargs=subplot_kwargs)
    assert ax[0].get_title() == "First Plot"


def test_gs_kwargs():
    gs_kwargs = {"width_ratios": [1, 2]}
    _fig, _ax, gs = make_plot_grid(2, 2, gs_kwargs=gs_kwargs, return_gridspec=True)
    assert gs.get_width_ratios() == [1, 2]


def test_figure_kwargs():
    figure_kwargs = {"figsize": (10, 5)}
    fig, _ax = make_plot_grid(2, 2, figure_kwargs=figure_kwargs)
    assert all(fig.get_size_inches() == [10, 5])


def test_3d_projection():
    subplot_kwargs = {0: {"projection": "3d"}}
    _fig, ax = make_plot_grid(2, 2, subplot_kwargs=subplot_kwargs)
    assert ax[0].name == "3d"


def test_sharez():
    subplot_kwargs = {0: {"projection": "3d"}}
    sharez = [(0, 1)]
    _fig, ax = make_plot_grid(2, 2, subplot_kwargs=subplot_kwargs, sharez=sharez)
    with pytest.raises(AttributeError):
        ax[1].get_proj()
    assert ax[0].get_proj().shape == (4, 4)


def test_large_grid():
    fig, ax = make_plot_grid(10, 10)
    assert len(ax) == 100
    assert isinstance(fig, plt.Figure)
    for axis in ax:
        assert isinstance(axis, plt.Axes)


def test_non_square_grid():
    fig, ax = make_plot_grid(3, 2)
    assert len(ax) == 6
    assert isinstance(fig, plt.Figure)
    for axis in ax:
        assert isinstance(axis, plt.Axes)


def test_subplot_spanning_multiple_grid_spaces():
    ranges = [(0, 1, 0, 1), (2, 2, 0, 1)]
    fig, ax = make_plot_grid(3, 2, ranges=ranges)
    assert len(ax) == 2
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax[0], plt.Axes)
    assert isinstance(ax[1], plt.Axes)


def test_sharex_mixed():
    sharex = True
    ranges = [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 1, 1)]
    _fig, ax = make_plot_grid(2, 2, sharex=sharex, ranges=ranges)
    assert len(ax) == 4
    for axis in ax[1:]:
        assert axis.get_shared_x_axes().joined(ax[0], axis)


def test_incomplete_ranges():
    ranges = [(0, 0, 0, 0), (1, 1, 1, 1)]
    fig, ax = make_plot_grid(2, 2, ranges=ranges)
    assert len(ax) == 2
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax[0], plt.Axes)
    assert isinstance(ax[1], plt.Axes)


def test_invalid_input_type():
    with pytest.raises(TypeError):
        make_plot_grid("2", "2")


def test_mixed_2d_3d_projections():
    subplot_kwargs = {0: {"projection": "3d"}}
    _fig, ax = make_plot_grid(2, 2, subplot_kwargs=subplot_kwargs)
    assert ax[0].name == "3d"
    for axis in ax[1:]:
        assert axis.name == "rectilinear"


def test_subplot_labels():
    subplot_kwargs = {0: {"title": "Title", "xlabel": "X Axis", "ylabel": "Y Axis"}}
    _fig, ax = make_plot_grid(2, 2, subplot_kwargs=subplot_kwargs)
    assert ax[0].get_title() == "Title"
    assert ax[0].get_xlabel() == "X Axis"
    assert ax[0].get_ylabel() == "Y Axis"


def test_height_ratios():
    gs_kwargs = {"height_ratios": [1, 3]}
    _fig, _ax, gs = make_plot_grid(2, 2, gs_kwargs=gs_kwargs, return_gridspec=True)
    assert gs.get_height_ratios() == [1, 3]


def test_empty_ranges():
    ranges = []
    _fig, ax = make_plot_grid(2, 2, ranges=ranges)
    assert len(ax) == 0


def test_very_large_grid():
    fig, ax = make_plot_grid(32, 32)
    assert len(ax) == 10000
    assert isinstance(fig, plt.Figure)
    for axis in ax:
        assert isinstance(axis, plt.Axes)


def test_single_row_grid():
    fig, ax = make_plot_grid(1, 5)
    assert len(ax) == 5
    assert isinstance(fig, plt.Figure)
    for axis in ax:
        assert isinstance(axis, plt.Axes)


def test_single_column_grid():
    fig, ax = make_plot_grid(5, 1)
    assert len(ax) == 5
    assert isinstance(fig, plt.Figure)
    for axis in ax:
        assert isinstance(axis, plt.Axes)


def test_mixed_invalid_ranges():
    ranges = [(0, 0, 0, 1), (0, 0, 1, 2)]
    with pytest.raises(ValueError, match="Provided ranges cause overlapping subplots"):
        make_plot_grid(2, 2, ranges=ranges)


def test_make_plot_grid_overlapping_range_raises():
    ranges = [(0, 1, 0, 0), (1, 2, 0, 0)]  # overlaps at row 1, col 0
    with pytest.raises(ValueError, match="overlapping subplots"):
        make_plot_grid(3, 3, ranges=ranges)


@pytest.mark.parametrize(
    ("sharex", "sharey"),
    [
        (True, False),
        (False, True),
        ([(0, 1)], [(1, 2)]),
    ],
)
def test_make_plot_grid_with_shared_axes(sharex, sharey):
    ranges = [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0)]
    _fig, ax = make_plot_grid(3, 3, ranges=ranges, sharex=sharex, sharey=sharey)
    assert len(ax) == 3


def test_subplot_kwargs_and_sharez():
    # Two subplots, the second is 3D and shares its Z-axis with the first
    ranges = [(0, 0, 0, 1), (1, 1, 0, 1)]
    _fig, ax = make_plot_grid(
        2,
        2,
        ranges=ranges,
        sharez=[(0, 1)],
        subplot_kwargs={1: {"projection": "3d"}},
    )
    assert len(ax) == 2
    # First axis is 2D, second is 3D
    assert hasattr(ax[1], "get_zlim")
    # They should share the same z-axis group
    shared = ax[1].get_shared_z_axes()
    assert shared.joined(ax[0], ax[1])


def test_subplot_kwargs_override_default_facecolor():
    # subplot_kwargs should override default facecolor
    ranges = [(0, 0, 0, 0)]
    _fig, ax = make_plot_grid(
        1,
        1,
        ranges=ranges,
        subplot_kwargs={0: {"facecolor": "red"}},
    )
    facecolor = ax[0].get_facecolor()
    # RGBA for red should be (1.0, 0.0, 0.0, 1.0)
    assert pytest.approx(facecolor) == (1.0, 0.0, 0.0, 1.0)
