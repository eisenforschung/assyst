import unittest
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings
from ase import Atoms

from assyst.crystals import Formulas, sample, _get_real_spacegroup


class TestFormulas(unittest.TestCase):

    def test_range(self):
        f = Formulas.range("Cu", 1, 4)
        self.assertEqual(len(f), 3, msg="Length of range('Cu', 1, 4) should be 3")
        self.assertEqual(f[0], {"Cu": 1}, msg="First element should be {'Cu': 1}")
        self.assertEqual(f[1], {"Cu": 2}, msg="Second element should be {'Cu': 2}")
        self.assertEqual(f[2], {"Cu": 3}, msg="Third element should be {'Cu': 3}")
        self.assertEqual(f.elements, {"Cu"}, msg="Elements should be {'Cu'}")

    def test_binary_range(self):
        f = Formulas.range(("Cu", "Ag"), 1, 3)
        self.assertEqual(f.elements, {"Cu", "Ag"}, msg="Elements should contain all given elements: {'Cu', 'Ag'}")
        self.assertEqual(
            Formulas.range(("Cu", "Ag"), 1, 3),
            Formulas.range("Cu", 1, 3) * Formulas.range("Ag", 1, 3),
            msg="range called with 2 elements should match outer product"
        )

    def test_addition(self):
        f1 = Formulas.range("Cu", 1, 3)
        f2 = Formulas.range("Cu", 3, 5)
        combined = f1 + f2
        self.assertIsInstance(combined, Formulas, msg="Result of addition should be a Formulas instance")
        self.assertEqual(len(combined), 4, msg="Combined length should be 4")
        self.assertEqual(combined[0], {"Cu": 1}, msg="First element after addition should be {'Cu': 1}")
        self.assertEqual(combined[-1], {"Cu": 4}, msg="Last element after addition should be {'Cu': 4}")

    def test_or_operator(self):
        cu = Formulas.range("Cu", 1, 3)
        ag = Formulas.range("Ag", 1, 3)
        result = cu | ag
        self.assertIsInstance(result, Formulas, msg="Result of | operation should be a Formulas instance")
        self.assertIn({"Cu": 1, "Ag": 1}, result, msg="Result should contain {'Cu': 1, 'Ag': 1}")
        self.assertIn({"Cu": 2, "Ag": 2}, result, msg="Result should contain {'Cu': 2, 'Ag': 2}")

        with self.assertRaises(AssertionError, msg="Should raise AssertionError for overlapping elements"):
            _ = cu | cu

    def test_mul_operator(self):
        cu = Formulas.range("Cu", 1, 3)
        ag = Formulas.range("Ag", 1, 3)
        result = cu * ag
        expected = [
            {"Cu": 1, "Ag": 1},
            {"Cu": 1, "Ag": 2},
            {"Cu": 2, "Ag": 1},
            {"Cu": 2, "Ag": 2}
        ]
        self.assertEqual(len(result), 4, msg="Outer product should contain 4 combinations")
        for r in expected:
            self.assertIn(r, result, msg=f"Expected combination {r} missing in result")

        with self.assertRaises(AssertionError, msg="Should raise AssertionError for overlapping elements"):
            _ = cu * cu

    def test_sequence_protocol(self):
        f = Formulas.range("Cu", 1, 3)
        self.assertIsInstance(f[0], dict, msg="Items in Formulas should be dicts")
        self.assertEqual(len(f), 2, msg="Length of range('Cu', 1, 3) should be 2")

    def test_trim(self):
        f = Formulas.range(("Cu", "Ag"), 10)
        for fi in f.trim():
            self.assertNotEqual(sum(fi.values()), 0,
                                msg="Trim called with no arguments should remove zero sizes formulas.")

        for fi in f.trim(min_atoms=3):
            self.assertGreaterEqual(sum(fi.values()), 3,
                                    msg="min_atoms should remove all formulas with less atoms.")

        for fi in f.trim(max_atoms=8):
            self.assertLessEqual(sum(fi.values()), 8,
                                 msg="max_atoms should remove all formulas with more atoms.")

def make_mock_atoms():
    atoms = MagicMock(spec=Atoms)
    atoms.info = {}
    return atoms


class TestSampleGrid(unittest.TestCase):
    """sample() walks the (formula, spacegroup) grid -- in the default or the shuffled order -- drawing exactly
    one structure per pyxtal() call. Both orders visit the same grid and share the same implementation, so most
    behaviour here must hold regardless of `shuffle`; a few tests at the end are specific to one order or the
    other."""

    formulas = Formulas.range("Cu", 1, 4)  # Cu1, Cu2, Cu3
    spacegroups = [1, 2, 3]

    def setUp(self):
        # pyxtal() is called once per grid point and returns a single Atoms, not a list
        pyxtal_patch = patch("assyst.crystals.pyxtal", side_effect=lambda *_, **__: make_mock_atoms())
        # sniffing the symmetry of the mocked structures is not possible
        spacegroup_patch = patch("assyst.crystals._get_real_spacegroup", return_value=1)
        self.mock_pyxtal = pyxtal_patch.start()
        spacegroup_patch.start()
        self.addCleanup(pyxtal_patch.stop)
        self.addCleanup(spacegroup_patch.stop)

    def drawn_grid_points(self):
        """The (species, num_ions, group) triples requested from pyxtal, in the order they were requested."""
        return [(call.args[1], call.args[2], call.args[0]) for call in self.mock_pyxtal.call_args_list]

    def test_covers_full_grid(self):
        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                self.mock_pyxtal.reset_mock()
                results = list(sample(self.formulas, self.spacegroups, shuffle=shuffle, rng=0))
                self.assertEqual(
                    len(results), 9, msg="Every point of the formula x spacegroup grid should yield a structure"
                )
                self.assertEqual(
                    set(self.drawn_grid_points()),
                    {(("Cu",), (n,), g) for n in (1, 2, 3) for g in self.spacegroups},
                    msg="sample() should visit every grid point exactly once",
                )

    def test_one_structure_per_call(self):
        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                self.mock_pyxtal.reset_mock()
                it = sample(self.formulas, self.spacegroups, shuffle=shuffle, rng=0)
                next(it)
                self.assertEqual(
                    self.mock_pyxtal.call_count, 1,
                    msg="The first structure should cost a single pyxtal call, not a whole formula",
                )

    def test_max_structures(self):
        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                self.mock_pyxtal.reset_mock()
                results = list(sample(self.formulas, self.spacegroups, max_structures=4, shuffle=shuffle, rng=0))
                self.assertEqual(len(results), 4, msg="Should not generate more than max_structures=4")
                self.assertEqual(
                    self.mock_pyxtal.call_count, 4, msg="Should not generate structures that are not yielded"
                )

    def test_min_atoms(self):
        formulas = (Formulas.range("Cu", 1, 10), Formulas.range("Cu", 10) * Formulas.range("Ag", 10))
        for shuffle in (False, True):
            for f in formulas:
                with self.subTest(shuffle=shuffle, formulas=f):
                    self.mock_pyxtal.reset_mock()
                    list(sample(f, [1], min_atoms=5, shuffle=shuffle, rng=0))
                    for _, num_ions, _ in self.drawn_grid_points():
                        self.assertLessEqual(
                            5, sum(num_ions), "sample tried to call pyxtal with fewer atoms than it should have."
                        )

    def test_max_atoms(self):
        formulas = (Formulas.range("Cu", 1, 10), Formulas.range("Cu", 10) * Formulas.range("Ag", 10))
        for shuffle in (False, True):
            for f in formulas:
                with self.subTest(shuffle=shuffle, formulas=f):
                    self.mock_pyxtal.reset_mock()
                    list(sample(f, [1], max_atoms=5, shuffle=shuffle, rng=0))
                    for _, num_ions, _ in self.drawn_grid_points():
                        self.assertLessEqual(
                            sum(num_ions), 5, "sample tried to call pyxtal with more atoms than it should have."
                        )

    def test_default_order_is_formula_major(self):
        list(sample(self.formulas, self.spacegroups, rng=0))
        formulas_seen = [num_ions for _, num_ions, _ in self.drawn_grid_points()]
        self.assertEqual(
            formulas_seen, [(1,)] * 3 + [(2,)] * 3 + [(3,)] * 3,
            msg="Without shuffle, sample() should exhaust one formula before moving to the next",
        )

    def test_shuffle_mixes_formulas(self):
        list(sample(self.formulas, self.spacegroups, shuffle=True, rng=0))
        first_formulas = {num_ions for _, num_ions, _ in self.drawn_grid_points()[:3]}
        self.assertGreater(
            len(first_formulas), 1,
            msg="The start of a shuffled stream should mix formulas, not exhaust the first one",
        )

    def test_shuffle_order_depends_on_seed(self):
        list(sample(self.formulas, self.spacegroups, shuffle=True, rng=0))
        first = self.drawn_grid_points()
        self.mock_pyxtal.reset_mock()

        list(sample(self.formulas, self.spacegroups, shuffle=True, rng=0))
        self.assertEqual(first, self.drawn_grid_points(), msg="The same seed should give the same order")
        self.mock_pyxtal.reset_mock()

        list(sample(self.formulas, self.spacegroups, shuffle=True, rng=1))
        self.assertNotEqual(first, self.drawn_grid_points(), msg="A different seed should give a different order")


class TestSampleIncompatibleGridPoints(unittest.TestCase):
    def test_incompatible_grid_points_are_skipped(self):
        """Mg1 cannot be placed in group 194; sample() should skip the pair instead of raising."""
        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                self.assertEqual(list(sample([{"Mg": 1}], [194], shuffle=shuffle, rng=0)), [])
                self.assertEqual(len(list(sample([{"Mg": 1}], [194, 1], shuffle=shuffle, rng=0))), 1)


class TestSampleSpaceGroupsArguments(unittest.TestCase):
    def test_invalid_dim(self):
        with self.assertRaises(ValueError):
            list(sample(Formulas.range("Cu", 1, 2), dim=4))

    def test_invalid_spacegroups(self):
        with self.assertRaises(ValueError):
            list(sample(Formulas.range("Cu", 1, 2), spacegroups=[0, 1]))
        with self.assertRaises(ValueError):
            list(sample(Formulas.range("Cu", 1, 2), spacegroups=[231]))

    def test_invalid_tolerance(self):
        with self.assertRaises(ValueError):
            list(sample(Formulas.range("Cu", 1, 2), tolerance="invalid"))

    @patch("assyst.crystals.pyxtal")
    def test_empty_stoichiometry(self, mock_pyxtal):
        mock_pyxtal.side_effect = lambda *_, **__: make_mock_atoms()
        formulas = Formulas(atoms=({},))
        results = list(sample(formulas, [1]))
        self.assertEqual(len(results), 0)
        mock_pyxtal.assert_not_called()

    @patch("assyst.crystals.pyxtal")
    def test_all_zero_formula_skipped_after_trim(self, mock_pyxtal):
        """min_atoms=0 lets an all-zero formula survive Formulas.trim(); _stoichiometries() must still skip it."""
        mock_pyxtal.side_effect = lambda *_, **__: make_mock_atoms()
        results = list(sample([{"Cu": 0}], [1], min_atoms=0))
        self.assertEqual(len(results), 0)
        mock_pyxtal.assert_not_called()

    @patch("assyst.crystals._get_real_spacegroup", return_value=1)
    @patch("assyst.crystals.pyxtal")
    def test_empty_dict_tolerance(self, mock_pyxtal, mock_spacegroup):
        mock_pyxtal.side_effect = lambda *_, **__: make_mock_atoms()
        list(sample(Formulas.range("Cu", 1, 2), [1], tolerance={}))
        self.assertIsNone(mock_pyxtal.call_args.kwargs['tm'])

    @patch("assyst.crystals._get_real_spacegroup", return_value=1)
    @patch("assyst.crystals.pyxtal")
    def test_distance_filter_tolerance(self, mock_pyxtal, mock_spacegroup):
        from assyst.filters import DistanceFilter
        mock_pyxtal.side_effect = lambda *_, **__: make_mock_atoms()
        list(sample(Formulas.range("Cu", 1, 2), [1], tolerance=DistanceFilter({'Cu': 1.0})))
        self.assertIsNotNone(mock_pyxtal.call_args.kwargs['tm'])


@settings(deadline=None, max_examples=50)
@given(st.integers(1, 230), st.booleans())
def test_spacegroup_info(group, shuffle):
    """sample() should add two fields to Atoms.info describing the requested and actual space group for
    each structure, in either traversal order."""
    for atoms in sample([{"Cu": 4}], [group], shuffle=shuffle):
        assert "requested spacegroup" in atoms.info and "spacegroup" in atoms.info, \
            "sample() does not supply spacegroup metadata!"
        assert atoms.info["requested spacegroup"] == group \
            and atoms.info["spacegroup"] == _get_real_spacegroup(atoms), \
            "sample() supplies wrong spacegroup metadata!"


if __name__ == "__main__":
    unittest.main()
