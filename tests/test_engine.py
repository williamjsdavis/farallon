"""Physical/representation contracts for the bounded geology engine."""

import copy
import unittest

import numpy as np

from geology.engine import evaluate_points, render_map, render_volume
from geology.history import HistoryError, history_to_program, parse_history


BASE = "strata(levels=[-1, -0.5, 0], units=[1, 2, 3, 4])"


class HistoryTests(unittest.TestCase):
    def test_roundtrip(self):
        program = BASE + "\nanticline(x=1, y=2, azimuth=135, uplift=1)\nerode(level=0)"
        history = parse_history(program)
        self.assertEqual(parse_history(history_to_program(history)), history)

    def test_syncline_roundtrip_and_nonnegative_magnitude_bounds(self):
        history = parse_history(BASE + "\nsyncline(x=1,y=2,uplift=1.2,plunge_se=5)\nerode(level=0)")
        self.assertEqual(parse_history(history_to_program(history)), history)
        for parameter in ["uplift=-.1", "uplift=10.1", "dip_ne=86", "hinge=0", "plunge_se=61"]:
            with self.subTest(parameter=parameter), self.assertRaises(HistoryError):
                parse_history(BASE + f"\nsyncline({parameter})")

    def test_program_cannot_execute_python(self):
        for fragment in ["import os", "x=3", "__import__('os').system('touch /tmp/bad')",
                         "anticline(uplift=1+2)", "anticline(**{})", "anticline(float('nan'))",
                         "anticline(uplift=True)", "anticline(uplift=1e999)",
                         "anticline(uplift=[x for x in []])", "anticline(uplift=1, uplift=2)"]:
            with self.subTest(fragment=fragment), self.assertRaises(HistoryError):
                parse_history(BASE + "\n" + fragment)

    def test_stratigraphy_and_unsupported_chronology(self):
        for program in ["strata(levels=[0,-1], units=[1,2,3])",
                        "strata(levels=[0], units=[1,1])",
                        "strata(levels=[0], units=[0,1])",
                        "strata(levels=[0], units=[1])",
                        BASE + "\nerode()\nanticline()",
                        BASE + "\nanticline(dip_ne=90)"]:
            with self.subTest(program=program), self.assertRaises(HistoryError):
                parse_history(program)


class EngineTests(unittest.TestCase):
    def test_layers_and_erosion(self):
        points = [[0, 0, -2], [0, 0, -1], [0, 0, -.75], [0, 0, -.25], [0, 0, .1]]
        np.testing.assert_array_equal(evaluate_points(BASE + "\nerode(level=0)", points), [1, 2, 2, 3, 0])

    def test_zero_magnitude_events_are_identity(self):
        points = np.random.default_rng(4).uniform(-2, 2, size=(1000, 3))
        expected = evaluate_points(BASE, points)
        for event in ["anticline(uplift=0)", "syncline(uplift=0)", "tilt(angle=0)", "fault(slip=0)"]:
            np.testing.assert_array_equal(evaluate_points(BASE + "\n" + event, points), expected)

    def test_syncline_lowers_strata_and_restores_points_upward(self):
        u, radius = .2, .1
        displacement = 1 - (np.sqrt(u * u + radius * radius) - radius)
        original_z = np.array([-1.2, -.8, -.3, .2])
        program = BASE + "\nsyncline(azimuth=0,uplift=1,dip_ne=45,dip_sw=20,hinge=.1,plunge_nw=0,plunge_se=0)"
        points = np.column_stack([-np.full(4, u), np.zeros(4), original_z - displacement])
        np.testing.assert_array_equal(evaluate_points(program, points), [1, 2, 3, 4])
        # With upright horizontal layers and a flat cut, opposite signed folds
        # expose older versus younger cores, while the far field is unchanged.
        strata = "strata(levels=[-.5,.5],units=[1,2,3])"
        params = "(azimuth=0,uplift=1,plunge_nw=0,plunge_se=0)"
        np.testing.assert_array_equal(render_map(strata + "\nanticline" + params, [0, 20], [0]), [[1, 2]])
        np.testing.assert_array_equal(render_map(strata + "\nsyncline" + params, [0, 20], [0]), [[3, 2]])

    def test_matching_upward_and_downward_folds_cancel(self):
        params = "(x=.2,y=.3,uplift=.8,azimuth=135,dip_ne=60,dip_sw=20,plunge_nw=30,plunge_se=10)"
        points = np.random.default_rng(15).uniform(-2, 2, size=(1000, 3))
        history = BASE + "\nanticline" + params + "\nsyncline" + params
        np.testing.assert_array_equal(evaluate_points(history, points), evaluate_points(BASE, points))

    def test_fold_forward_and_inverse(self):
        # A hand-computed point in the planar-ish NE limb is raised through a
        # rounded hinge; restoring it must recover the original stratigraphy.
        u, uplift, radius, dip = .2, 1.0, .1, np.deg2rad(45)
        displacement = uplift - np.tan(dip) * (np.sqrt(u * u + radius * radius) - radius)
        original_z = np.array([-1.2, -.8, -.3, .2])
        program = BASE + "\nanticline(azimuth=0,uplift=1,dip_ne=45,dip_sw=20,hinge=.1,plunge_nw=0,plunge_se=0)"
        points = np.column_stack([-np.full(4, u), np.zeros(4), original_z + displacement])
        np.testing.assert_array_equal(evaluate_points(program, points), [1, 2, 3, 4])

    def test_asymmetry_and_nose_closure(self):
        program = "strata(levels=[-.5],units=[1,2])\nanticline(azimuth=135,uplift=1,dip_ne=60,dip_sw=20,hinge=.1,nw_length=.1,se_length=.1,plunge_nw=30,plunge_se=0)"
        ne = np.array([1, 1]) / np.sqrt(2)
        se = np.array([1, -1]) / np.sqrt(2)
        points = [[*(.5 * ne), 0], [*(-.5 * ne), 0], [*(-2 * se), 0], [*(2 * se), 0]]
        # Steep NE side exposes younger rock at equal horizontal distance;
        # NW end closes, while the unplunging SE end stays in the old core.
        np.testing.assert_array_equal(evaluate_points(program, points), [2, 1, 2, 1])

    def test_fault_offsets_older_units(self):
        program = BASE + "\nfault(azimuth=0,dip=60,slip=.5)"
        points = np.array([[1, 0, -.75], [-1, 0, -.75]])
        np.testing.assert_array_equal(evaluate_points(program, points), [3, 2])

    def test_intrusion_timing_changes_displacement(self):
        intrusion = "intrusion(x=1,y=0,z=-.75,rx=.1,ry=.1,rz=.1,unit=8)"
        fault = "fault(azimuth=0,dip=60,slip=.5)"
        moved = [1 + .5 * np.cos(np.deg2rad(60)), 0, -.75 - .5 * np.sin(np.deg2rad(60))]
        original = [1, 0, -.75]
        before = evaluate_points(BASE + "\n" + intrusion + "\n" + fault, [moved, original])
        after = evaluate_points(BASE + "\n" + fault + "\n" + intrusion, [moved, original])
        self.assertEqual(before[0], 8)
        self.assertNotEqual(before[1], 8)
        self.assertNotEqual(after[0], 8)
        self.assertEqual(after[1], 8)

    def test_volume_map_points_agree_and_y_order_is_explicit(self):
        history = parse_history(BASE + "\nanticline(x=.2,y=.3,uplift=.8)\nerode(level=.5)")
        original = copy.deepcopy(history)
        bounds, resolution = [-1, 1, -1, 1, -1.5, .5], (9, 7, 5)
        volume = render_volume(history, bounds, resolution)
        xs = np.linspace(-1 + 1 / 9, 1 - 1 / 9, 9)
        ys = np.linspace(-1 + 1 / 7, 1 - 1 / 7, 7)
        zs = np.linspace(-1.5 + .2, .5 - .2, 5)
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        np.testing.assert_array_equal(volume, evaluate_points(history, points).reshape(5, 7, 9))
        surface = render_map(history, xs, ys[::-1], terrain=zs[2] + 1e-8)
        np.testing.assert_array_equal(surface, volume[2, ::-1])
        self.assertEqual(history, original)
        self.assertEqual(volume.dtype, np.uint8)

    def test_external_surface_replaces_final_erosion_and_exposes_positive_elevations(self):
        history = BASE + "\nerode(level=-.25)"
        # Omitting terrain retains the original clipping behavior. Supplying
        # terrain declares the actual erosion surface, even above old z=0.
        np.testing.assert_array_equal(render_map(history, [0, 1], [1, 0]), np.zeros((2, 2), dtype=np.uint8))
        np.testing.assert_array_equal(render_map(history, [0, 1], [1, 0], terrain=1), np.full((2, 2), 4, dtype=np.uint8))
        volume = render_volume(history, [0, 2, 0, 2, 0, 2], 2, terrain=1)
        np.testing.assert_array_equal(volume[0], np.full((2, 2), 4, dtype=np.uint8))
        np.testing.assert_array_equal(volume[1], np.zeros((2, 2), dtype=np.uint8))
        # A fixed surface also clips histories that have no erode call.
        np.testing.assert_array_equal(volume, render_volume(BASE, [0, 2, 0, 2, 0, 2], 2, terrain=1))

    def test_relief_controls_map_exposure_and_volume_air_with_explicit_orientation(self):
        program = BASE + "\nanticline(x=.2,y=.3,uplift=.8)\nerode(level=0)"
        history = parse_history(program)
        original = copy.deepcopy(history)
        uneroded = copy.deepcopy(history)
        uneroded["events"].pop()
        bounds, resolution = [-1, 1, -1, 1, -1, 1], (9, 7, 5)
        xs = -1 + (np.arange(9) + .5) * 2 / 9
        ys = -1 + (np.arange(7) + .5) * 2 / 7
        zs = -1 + (np.arange(5) + .5) * 2 / 5
        xx, yy = np.meshgrid(xs, ys)
        terrain = .213 + .3 * xx - .47 * yy
        self.assertGreater(terrain.max(), 0)
        self.assertLess(terrain.min(), 0)

        surface_points = np.column_stack([xx.ravel(), yy.ravel(), terrain.ravel() - 1e-8])
        expected_map = evaluate_points(uneroded, surface_points).reshape(7, 9)
        np.testing.assert_array_equal(render_map(history, xs, ys, terrain), expected_map)
        np.testing.assert_array_equal(render_map(history, xs, ys[::-1], terrain[::-1]), expected_map[::-1])

        zz, yyy, xxx = np.meshgrid(zs, ys, xs, indexing="ij")
        points = np.column_stack([xxx.ravel(), yyy.ravel(), zz.ravel()])
        expected_volume = evaluate_points(uneroded, points).reshape(5, 7, 9)
        expected_volume[zz > terrain[None, :, :]] = 0
        volume = render_volume(history, bounds, resolution, terrain=terrain)
        np.testing.assert_array_equal(volume, expected_volume)
        self.assertTrue(np.any((volume > 0) & (zz > 0)))
        self.assertTrue(np.any((volume == 0) & (zz < 0)))
        self.assertEqual(history, original)

    def test_terrain_requires_finite_heights_at_matching_sample_locations(self):
        for terrain in [float("nan"), float("inf"), [[0, 1]], np.zeros((2, 2, 1)), [[0, np.nan], [0, 0]]]:
            with self.subTest(terrain=terrain):
                with self.assertRaises(ValueError):
                    render_map(BASE, [0, 1], [1, 0], terrain=terrain)
                with self.assertRaises(ValueError):
                    render_volume(BASE, [0, 2, 0, 2, 0, 2], 2, terrain=terrain)


if __name__ == "__main__":
    unittest.main()
