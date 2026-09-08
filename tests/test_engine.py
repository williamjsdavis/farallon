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
        for event in ["anticline(uplift=0)", "tilt(angle=0)", "fault(slip=0)"]:
            np.testing.assert_array_equal(evaluate_points(BASE + "\n" + event, points), expected)

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

    def test_air_at_known_terrain_is_not_hidden(self):
        np.testing.assert_array_equal(render_map(BASE + "\nerode(level=0)", [0, 1], [1, 0], terrain=1), np.zeros((2, 2), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
