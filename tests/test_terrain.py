"""Independent coordinate and interpolation checks for fixed DEM sampling."""
import numpy as np
import pytest

from geology.terrain import TerrainGrid


def plane(xs, ys):
    return 7 + 2.5 * np.asarray(xs)[None, :] - 1.5 * np.asarray(ys)[:, None]


@pytest.mark.parametrize("flip_x,flip_y", [(False, False), (True, False), (False, True), (True, True)])
def test_non_square_uneven_grid_reproduces_plane_in_any_axis_order(flip_x, flip_y):
    xs, ys = np.array([0., 2., 5.]), np.array([-1., 3.])
    if flip_x:
        xs = xs[::-1]
    if flip_y:
        ys = ys[::-1]
    grid = TerrainGrid(plane(xs, ys), xs, ys)
    query_x, query_y = [4., .5, 2.], [2., -.5]
    sampled = grid.sample(query_x, query_y)
    assert sampled.shape == (2, 3)
    assert sampled.dtype == np.float64
    np.testing.assert_allclose(sampled, plane(query_x, query_y), rtol=0, atol=1e-12)


def test_outer_half_cell_and_far_outside_queries_clamp_to_edge():
    xs, ys = [0., 2., 5.], [-1., 3.]
    grid = TerrainGrid(plane(xs, ys), xs, ys)
    query_x = [-100, -.25, 0, 2.5, 5, 5.25, 100]
    query_y = [-100, -1.25, -1, 1, 3, 3.25, 100]
    expected = plane(np.clip(query_x, 0, 5), np.clip(query_y, -1, 3))
    np.testing.assert_allclose(grid.sample(query_x, query_y), expected, rtol=0, atol=1e-12)


def test_bilinear_cross_term_and_exact_source_samples():
    xs, ys = np.array([0., 2., 5.]), np.array([-1., 3.])
    heights = xs[None, :] * ys[:, None]
    grid = TerrainGrid(heights, xs, ys)
    qx, qy = np.array([.5, 1.2, 4.]), np.array([-.5, 2.])
    np.testing.assert_allclose(grid.sample(qx, qy), qx[None, :] * qy[:, None], atol=1e-12)
    np.testing.assert_array_equal(grid.sample(xs, ys), heights)


def test_singleton_axes_remain_constant_outside_measured_coordinate():
    grid = TerrainGrid([[2], [6]], [4], [0, 2])
    np.testing.assert_array_equal(grid.sample([-100, 4, 100], [1]), [[4, 4, 4]])
    grid = TerrainGrid([[2, 6]], [0, 2], [4])
    np.testing.assert_array_equal(grid.sample([1], [-100, 4, 100]), [[4], [4], [4]])
    grid = TerrainGrid([[3]], [0], [0])
    np.testing.assert_array_equal(grid.sample([-1, 1], [1, -1]), np.full((2, 2), 3.))


def test_source_mutation_cannot_change_fixed_terrain():
    heights, xs, ys = np.array([[1., 2.], [3., 4.]]), np.array([0., 1.]), np.array([0., 1.])
    grid = TerrainGrid(heights, xs, ys)
    heights[:] = 100
    xs[:] = 100
    ys[:] = 100
    assert grid.sample([.5], [.5])[0, 0] == 2.5
    for array in (grid.heights_km, grid.xs, grid.ys):
        assert not array.flags.writeable


@pytest.mark.parametrize("heights,xs,ys", [
    ([[1, 2]], [0, 0], [0]),
    ([[1, 2, 3]], [0, 2, 1], [0]),
    ([[1], [2], [3]], [0], [0, 2, 1]),
    ([[1, 2]], [0, np.inf], [0]),
    ([[1, np.nan]], [0, 1], [0]),
    ([[1, 2]], [0, 1], [np.nan]),
    ([[1, 2]], [[0, 1]], [0]),
    ([[1, 2]], [0, 1], []),
    ([[1]], [0, 1], [0]),
])
def test_invalid_source_grids_are_rejected(heights, xs, ys):
    with pytest.raises(ValueError):
        TerrainGrid(heights, xs, ys)


@pytest.mark.parametrize("xs,ys", [([], [0]), ([0], []), ([np.nan], [0]), ([0], [np.inf]), ([[0]], [0])])
def test_invalid_query_coordinates_are_rejected(xs, ys):
    grid = TerrainGrid([[1]], [0], [0])
    with pytest.raises(ValueError):
        grid.sample(xs, ys)


def test_npz_loading_and_required_keys(tmp_path):
    path = tmp_path / "dem.npz"
    xs, ys = np.array([0., 2., 5.]), np.array([3., -1.])
    np.savez(path, heights_km=plane(xs, ys), xs=xs, ys=ys)
    grid = TerrainGrid.from_npz(path)
    np.testing.assert_allclose(grid.sample([1, 4], [2, 0]), plane([1, 4], [2, 0]))
    np.savez(path, xs=xs, ys=ys)
    with pytest.raises(ValueError, match="heights_km"):
        TerrainGrid.from_npz(path)
