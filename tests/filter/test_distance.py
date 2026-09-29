import unittest
import warnings
from itertools import product
from hypothesis import assume, given, strategies as st
import numpy as np
from ase import Atoms
from assyst.filters import DistanceFilter
from pyxtal.tolerance import Tol_matrix
from ase.data import atomic_numbers
from tests.strategies.strategies import elements


class TestDistanceFilter(unittest.TestCase):
    # radii of 2.5 keep the neighbour cutoff at about 5 Å, the value that used to be hardcoded
    cutoff_filter = DistanceFilter({'Cu': 2.5, 'Ag': 2.5})

    def test_element_wise_dist(self):
        """_element_wise_dist returns a dict with all element pair keys present in structure."""
        with self.subTest("unary"):
            structure = Atoms('Cu2', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.0, 0, 0)])
            pair = self.cutoff_filter._element_wise_dist(structure)
            self.assertIsInstance(pair, dict, msg="Return type should be dict")
            self.assertIn(('Cu', 'Cu'), pair, msg="Pair ('Cu', 'Cu') should be present in unary case")
        with self.subTest("binary"):
            structure = Atoms('CuAg', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.5, 0, 0)])
            pair = self.cutoff_filter._element_wise_dist(structure)
            self.assertIsInstance(pair, dict, msg="Return type should be dict")
            self.assertIn(('Cu', 'Cu'), pair, msg="Pair ('Cu', 'Cu') should be present in binary case")
            self.assertIn(('Ag', 'Cu'), pair, msg="Pair ('Ag', 'Cu') should be present in binary case")
            self.assertIn(('Ag', 'Ag'), pair, msg="Pair ('Ag', 'Ag') should be present in binary case")

    def test_element_wise_dist_values(self):
        """_element_wise_dist should return correct minimum pair distances."""
        structure = Atoms('Cu2', cell=[4, 4, 4], pbc=True, positions=[(0, 0, 0), (2.0, 0, 0)])
        pair = self.cutoff_filter._element_wise_dist(structure)
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


class TestDistanceFilterScalar(unittest.TestCase):
    """DistanceFilter built from a single number applies that radius to every element (#165)."""

    def test_call_rejects_close(self):
        """Scalar radius rejects a pair closer than twice the radius."""
        structure = Atoms('Cu2', cell=[20, 20, 20], pbc=True, positions=[(0, 0, 0), (0.2, 0, 0)])
        self.assertFalse(DistanceFilter(1.5)(structure), msg="d=0.2 < 2*1.5")

    def test_call_accepts_far(self):
        """Scalar radius accepts a pair farther than twice the radius."""
        structure = Atoms('Cu2', cell=[20, 20, 20], pbc=True, positions=[(0, 0, 0), (3.1, 0, 0)])
        self.assertTrue(DistanceFilter(1.5)(structure), msg="d=3.1 > 2*1.5")

    def test_call_threshold(self):
        """Scalar radius compares against 2*r for pairs just below and just above."""
        for d, expected in ((2.99, False), (3.01, True)):
            with self.subTest(d=d):
                structure = Atoms('Cu2', cell=[20, 20, 20], pbc=True, positions=[(0, 0, 0), (d, 0, 0)])
                self.assertIs(DistanceFilter(1.5)(structure), expected)

    def test_call_heterogeneous(self):
        """Scalar radius applies to mixed-element pairs."""
        structure = Atoms('CuAg', cell=[20, 20, 20], pbc=True, positions=[(0, 0, 0), (2.5, 0, 0)])
        self.assertFalse(DistanceFilter(1.5)(structure), msg="Cu-Ag d=2.5 < 2*1.5")
        self.assertTrue(DistanceFilter(1.2)(structure), msg="Cu-Ag d=2.5 > 2*1.2")

    def test_call_matches_mapping(self):
        """Scalar form gives the same verdict as the mapping form with equal radii."""
        structure = Atoms('CuAg2', cell=[10, 10, 10], pbc=True,
                          positions=[(0, 0, 0), (2.5, 0, 0), (5.5, 0, 0)])
        for r in (1.0, 1.2, 1.3, 1.6):
            with self.subTest(r=r):
                self.assertEqual(
                    DistanceFilter(r)(structure),
                    DistanceFilter({'Cu': r, 'Ag': r})(structure),
                )

    def test_to_tol_matrix(self):
        """Scalar form gives 2*r for any element pair, not the pyxtal prototype value."""
        tol = DistanceFilter(1.5).to_tol_matrix()
        self.assertIsInstance(tol, Tol_matrix)
        for a, b in (('Cu', 'Cu'), ('Cu', 'Ag'), ('H', 'U')):
            with self.subTest(pair=(a, b)):
                self.assertEqual(tol.get_tol(atomic_numbers[a], atomic_numbers[b]), 3.0)

    def test_to_tol_matrix_after_call(self):
        """Calling the filter does not change what to_tol_matrix returns afterwards."""
        filter = DistanceFilter(1.5)
        filter(Atoms('Cu2', cell=[20, 20, 20], pbc=True, positions=[(0, 0, 0), (4.0, 0, 0)]))
        tol = filter.to_tol_matrix()
        for a, b in (('Cu', 'Cu'), ('Cu', 'Ag'), ('Ag', 'Ag')):
            with self.subTest(pair=(a, b)):
                self.assertEqual(tol.get_tol(atomic_numbers[a], atomic_numbers[b]), 3.0)

    def test_mapping_missing_element_not_inserted(self):
        """Mapping form still treats absent elements as NaN (allowed) and does not grow the mapping."""
        structure = Atoms('CuAg', cell=[20, 20, 20], pbc=True, positions=[(0, 0, 0), (0.2, 0, 0)])
        radii = {'Cu': 1.5}
        self.assertTrue(DistanceFilter(radii)(structure), msg="Ag radius absent -> NaN -> pair allowed")
        self.assertEqual(radii, {'Cu': 1.5})


def _dimer(symbols, d, cell=40.0):
    return Atoms(symbols, cell=[cell] * 3, pbc=True, positions=[(0, 0, 0), (d, 0, 0)])


class TestDistanceFilterLargeRadii(unittest.TestCase):
    """Radii sums at or above 5 Å are enforced; the neighbour cutoff is not fixed at 5 Å (#164)."""

    def test_call_rejects_beyond_5(self):
        """Pairs closer than r_i + r_j = 6 Å are rejected, including those farther than 5 Å."""
        filter = DistanceFilter({'Cu': 3.0})
        for d in (4.9, 5.0, 5.01, 5.5, 5.99):
            with self.subTest(d=d):
                self.assertFalse(filter(_dimer('Cu2', d)), msg=f"d={d} < 2*3.0")

    def test_call_accepts_beyond_sum(self):
        """Pairs farther than r_i + r_j = 6 Å are accepted."""
        filter = DistanceFilter({'Cu': 3.0})
        for d in (6.01, 7.0):
            with self.subTest(d=d):
                self.assertTrue(filter(_dimer('Cu2', d)), msg=f"d={d} > 2*3.0")

    def test_call_heterogeneous(self):
        """A mixed pair whose radii sum past 5 Å is enforced."""
        filter = DistanceFilter({'Cu': 3.0, 'Ag': 2.8})
        self.assertFalse(filter(_dimer('CuAg', 5.5)), msg="Cu-Ag d=5.5 < 3.0+2.8")
        self.assertTrue(filter(_dimer('CuAg', 5.9, cell=60)), msg="Cu-Ag d=5.9 > 3.0+2.8")

    def test_call_periodic_image(self):
        """A single atom closer than r_i + r_j to its own periodic image is rejected."""
        structure = Atoms('Cu', cell=[5.5, 5.5, 5.5], pbc=True)
        self.assertFalse(DistanceFilter({'Cu': 3.0})(structure), msg="image d=5.5 < 2*3.0")
        self.assertTrue(DistanceFilter({'Cu': 2.7})(structure), msg="image d=5.5 > 2*2.7")

    def test_call_nan_radius_does_not_disable_others(self):
        """A NaN radius on one element does not switch off the check for the others, whatever the key order."""
        for radii in ({'Cu': np.nan, 'Ag': 3.0}, {'Ag': 3.0, 'Cu': np.nan}):
            with self.subTest(radii=list(radii)):
                filter = DistanceFilter(radii)
                self.assertFalse(filter(_dimer('Ag2', 1.0)), msg="Ag-Ag d=1.0 < 2*3.0")
                self.assertFalse(filter(_dimer('Ag2', 5.5)), msg="Ag-Ag d=5.5 < 2*3.0")
                self.assertTrue(filter(_dimer('Cu2', 1.0)), msg="Cu radius NaN allows Cu-Cu")

    def test_element_wise_dist_without_finite_radii(self):
        """Empty or all-NaN radii still give a finite neighbour cutoff and no numpy warning."""
        for radii in ({}, {'Cu': np.nan}, {'Cu': np.nan, 'Ag': np.nan}):
            with self.subTest(radii=radii), warnings.catch_warnings():
                warnings.simplefilter("error")
                pair = DistanceFilter(radii)._element_wise_dist(_dimer('Cu2', 2.0))
                self.assertIn(('Cu', 'Cu'), pair, msg="Cu-Cu at 2.0 must be within the default cutoff")
                self.assertAlmostEqual(pair[('Cu', 'Cu')], 2.0)

    def test_call_ignores_absent_large_radius(self):
        """A large radius for an element absent from the structure does not affect the verdict."""
        filter = DistanceFilter({'Cu': 1.0, 'Ag': 4.0})
        self.assertTrue(filter(_dimer('Cu2', 2.5)), msg="Cu-Cu d=2.5 > 2*1.0")
        self.assertFalse(filter(_dimer('Cu2', 1.5)), msg="Cu-Cu d=1.5 < 2*1.0")


radii = st.floats(1, allow_nan=False, allow_infinity=False)

@given(radii, radii, elements(), elements())
def test_to_tol_matrix(ra, rb, a, b):
    """to_tol_matrix returns a correct Tol_matrix object."""
    radii = {a: ra, b: rb}
    filter = DistanceFilter(radii)
    tol_matrix = filter.to_tol_matrix()

    assert isinstance(tol_matrix, Tol_matrix)

    for i, j in product((a, b), repeat=2):
        assert tol_matrix.get_tol(atomic_numbers[i], atomic_numbers[j]) == radii[i] + radii[j]


@given(radii, elements(), elements())
def test_to_tol_matrix_scalar(r, a, b):
    """to_tol_matrix of a scalar filter returns 2*r for any element pair."""
    tol_matrix = DistanceFilter(r).to_tol_matrix()
    assert tol_matrix.get_tol(atomic_numbers[a], atomic_numbers[b]) == 2 * r


@given(
    st.floats(0.5, 6.0, allow_nan=False, allow_infinity=False),
    st.floats(0.5, 15.0, allow_nan=False, allow_infinity=False),
)
def test_call_dimer_any_radius(r, d):
    """For any radius, a Cu dimer passes exactly when d >= 2*r."""
    assume(abs(d - 2 * r) > 1e-6)
    assert DistanceFilter({'Cu': r})(_dimer('Cu2', d)) == (d > 2 * r)


if __name__ == '__main__':
    unittest.main()
    test_to_tol_matrix()
