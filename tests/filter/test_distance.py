import unittest
import numpy as np
from ase import Atoms
from assyst.filters import DistanceFilter

class TestDistanceFilter(unittest.TestCase):
    def test_element_wise_dist(self):
        """_element_wise_dist returns a dict with all element pair keys present in structure."""
        with self.subTest("unary"):
            structure = Atoms('Cu2', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.0, 0, 0)])
            pair = DistanceFilter._element_wise_dist(structure)
            self.assertIsInstance(pair, dict, msg="Return type should be dict")
            self.assertIn(('Cu', 'Cu'), pair, msg="Pair ('Cu', 'Cu') should be present in unary case")
        with self.subTest("binary"):
            structure = Atoms('CuAg', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.5, 0, 0)])
            pair = DistanceFilter._element_wise_dist(structure)
            self.assertIsInstance(pair, dict, msg="Return type should be dict")
            self.assertIn(('Cu', 'Cu'), pair, msg="Pair ('Cu', 'Cu') should be present in binary case")
            self.assertIn(('Ag', 'Cu'), pair, msg="Pair ('Ag', 'Cu') should be present in binary case")
            self.assertIn(('Ag', 'Ag'), pair, msg="Pair ('Ag', 'Ag') should be present in binary case")

    def test_element_wise_dist_values(self):
        """_element_wise_dist should return correct minimum pair distances."""
        structure = Atoms('Cu2', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.0, 0, 0)])
        pair = DistanceFilter._element_wise_dist(structure)
        self.assertGreater(pair[('Cu', 'Cu')], 0, msg="Distance should be greater than zero")
        self.assertEqual(pair[('Cu', 'Cu')], 2.0, msg="Distance between Cu atoms is 2.0 units")

    def test_call_method(self):
        """__call__ returns True if all atomic pair distances exceed the sum of radii."""
        # Cu-Cu with radius 0.9, sum = 1.8, so distance 2.0 is valid
        structure = Atoms('Cu2', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.0, 0, 0)])
        filter = DistanceFilter({'Cu': 0.9})
        self.assertTrue(filter(structure), msg="Should be True: d=2.0 > 2*0.9")

    def test_call_method_false(self):
        """__call__ returns False if any atomic pair is closer than the sum of radii."""
        # Cu-Cu with radius 1.1, sum = 2.2, so distance 2.0 is invalid
        structure = Atoms('Cu2', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.0, 0, 0)])
        filter = DistanceFilter({'Cu': 1.1})
        self.assertFalse(filter(structure), msg="Should be False: d=2.0 < 2*1.1")

    def test_call_method_nan_radii(self):
        """__call__ returns True when radii for the relevant atom are NaN."""
        structure = Atoms('Cu2', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.0, 0, 0)])
        filter = DistanceFilter({'Cu': np.nan})
        self.assertTrue(filter(structure), msg="Should be True: radii is NaN")

    def test_call_method_empty_radii(self):
        """__call__ returns True when radii dictionary is empty."""
        structure = Atoms('Cu2', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.0, 0, 0)])
        filter = DistanceFilter({})
        self.assertTrue(filter(structure), msg="Should be True: radii dict is empty")

    def test_call_method_multiple_elements(self):
        """__call__ works for multi-element structures."""
        # Cu-Ag with radii 1.3 + 1.5 = 2.8, so all d > 2.8 is valid
        structure = Atoms('CuAg2', cell=[10, 10, 10], pbc=True,
                          positions=[(0, 0, 0), (3.0, 0, 0), (6.0, 0, 0)])
        filter = DistanceFilter({'Cu': 1.3, 'Ag': 1.5})
        self.assertTrue(filter(structure), msg="Should be True: all pair distances > radii sums")

    def test_call_method_multiple_elements_false(self):
        """__call__ returns False if any heterogeneous pair is too close (Cu-Ag < r_Cu + r_Ag)."""
        # Cu-Ag d = 2.5, sum = 2.8
        structure = Atoms('CuAg2', cell=[10, 10, 10], pbc=True,
                          positions=[(0, 0, 0), (2.5, 0, 0), (5.5, 0, 0)])
        filter = DistanceFilter({'Cu': 1.3, 'Ag': 1.5})
        self.assertFalse(filter(structure), msg="Should be False: d=2.5 < 1.3+1.5=2.8")

    def test_call_method_periodic_boundary(self):
        """__call__ correctly handles periodic boundary cases."""
        # One atom at 0, one at nearly cell edge, minimum image distance is 0.5
        structure = Atoms('Cu2', cell=[2.0, 2.0, 2.0], pbc=True, positions=[(0, 0, 0), (1.5, 0, 0)])
        filter = DistanceFilter({'Cu': 0.1})
        self.assertTrue(filter(structure), msg="minimal image d=0.5 > 2*0.1, so True")
        filter = DistanceFilter({'Cu': 0.3})
        self.assertFalse(filter(structure), msg="minimal image d=0.5 < 2*0.3")

if __name__ == '__main__':
    unittest.main()
