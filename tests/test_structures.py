import unittest
from unittest.mock import patch, MagicMock
from ase import Atoms

from assyst.structures import Formulas, sample_space_groups


class TestFormulas(unittest.TestCase):

    def test_unary_range(self):
        f = Formulas.unary_range("Cu", 1, 4)
        self.assertEqual(len(f), 3, msg="Length of unary_range('Cu', 1, 4) should be 3")
        self.assertEqual(f[0], {"Cu": 1}, msg="First element should be {'Cu': 1}")
        self.assertEqual(f[1], {"Cu": 2}, msg="Second element should be {'Cu': 2}")
        self.assertEqual(f[2], {"Cu": 3}, msg="Third element should be {'Cu': 3}")
        self.assertEqual(f.elements, {"Cu"}, msg="Elements should be {'Cu'}")

    def test_addition(self):
        f1 = Formulas.unary_range("Cu", 1, 3)
        f2 = Formulas.unary_range("Cu", 3, 5)
        combined = f1 + f2
        self.assertIsInstance(combined, Formulas, msg="Result of addition should be a Formulas instance")
        self.assertEqual(len(combined), 4, msg="Combined length should be 4")
        self.assertEqual(combined[0], {"Cu": 1}, msg="First element after addition should be {'Cu': 1}")
        self.assertEqual(combined[-1], {"Cu": 4}, msg="Last element after addition should be {'Cu': 4}")

    def test_or_operator(self):
        cu = Formulas.unary_range("Cu", 1, 3)
        ag = Formulas.unary_range("Ag", 1, 3)
        result = cu | ag
        self.assertIsInstance(result, Formulas, msg="Result of | operation should be a Formulas instance")
        self.assertIn({"Cu": 1, "Ag": 1}, result, msg="Result should contain {'Cu': 1, 'Ag': 1}")
        self.assertIn({"Cu": 2, "Ag": 2}, result, msg="Result should contain {'Cu': 2, 'Ag': 2}")

        with self.assertRaises(AssertionError, msg="Should raise AssertionError for overlapping elements"):
            _ = cu | cu

    def test_mul_operator(self):
        cu = Formulas.unary_range("Cu", 1, 3)
        ag = Formulas.unary_range("Ag", 1, 3)
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
        f = Formulas.unary_range("Cu", 1, 3)
        self.assertIsInstance(f[0], dict, msg="Items in Formulas should be dicts")
        self.assertEqual(len(f), 2, msg="Length of unary_range('Cu', 1, 3) should be 2")


def make_mock_atoms():
    atoms = MagicMock(spec=Atoms)
    atoms.info = {}
    return atoms

def make_pyxtal_mock_side_effect(n: int = 1):
    mock_atoms = make_mock_atoms()
    return mock_atoms, lambda *_, **__: [{"atoms": mock_atoms} for _ in range(n)]

class TestSampleSpaceGroups(unittest.TestCase):

    @patch("assyst.structures.pyxtal")
    def test_max_structures(self, mock_pyxtal):
        mock_atoms, mock_pyxtal.side_effect = make_pyxtal_mock_side_effect(5)

        f = Formulas.unary_range("Cu", 1, 3)
        results = list(sample_space_groups(f, max_structures=3))

        self.assertEqual(len(results), 3, msg="Should not generate more than max_structures=3")

    @patch("assyst.structures.pyxtal")
    def test_pyxtal_called_once_per_composition(self, mock_pyxtal):
        mock_atoms, mock_pyxtal.side_effect = make_pyxtal_mock_side_effect()

        # Define 3 compositions: Cu1, Cu2, Cu3
        formulas = Formulas.unary_range("Cu", 1, 4)  # 3 compositions

        results = list(sample_space_groups(formulas, max_structures=10))

        # We should get 3 results (since 1 per composition)
        self.assertEqual(len(results), 3, msg="Expected one structure per composition")
        self.assertEqual(mock_pyxtal.call_count, 3, msg="pyxtal should be called once per composition")

        expected_calls = [
            (('Cu',), (1,)),
            (('Cu',), (2,)),
            (('Cu',), (3,))
        ]
        actual_calls = [call.args for call in mock_pyxtal.call_args_list]

        for expected, actual in zip(expected_calls, actual_calls):
            self.assertEqual(
                    # actual called includes spacegroups that we did not include in the mock
                    actual[1:], expected,
                    msg=f"Expected pyxtal to be called with atom counts {expected[1]}, got {actual[1]}"
        )

    @patch("assyst.structures.pyxtal")
    def test_min_atoms(self, mock_pyxtal):
        mock_atoms, mock_pyxtal.side_effect = make_pyxtal_mock_side_effect()

        formulas = Formulas.unary_range("Cu", 1, 10)
        results = list(sample_space_groups(formulas, min_atoms=5))

        with self.subTest("unary"):
            for call in mock_pyxtal.call_args_list:
                self.assertLessEqual(5, sum(call.args[2]),
                    "sample_space_groups tried to call pyxtal with more atoms than it should have."
                )

        mock_pyxtal.reset_mock()

        formulas = Formulas.unary_range("Cu", 10) * Formulas.unary_range("Ag", 10)
        results = list(sample_space_groups(formulas, min_atoms=5))

        with self.subTest("binary"):
            for call in mock_pyxtal.call_args_list:
                self.assertLessEqual(5, sum(call.args[2]),
                    "sample_space_groups tried to call pyxtal with more atoms than it should have."
                )

    @patch("assyst.structures.pyxtal")
    def test_max_atoms(self, mock_pyxtal):
        mock_atoms, mock_pyxtal.side_effect = make_pyxtal_mock_side_effect()

        formulas = Formulas.unary_range("Cu", 1, 10)
        results = list(sample_space_groups(formulas, max_atoms=5))

        with self.subTest("unary"):
            for call in mock_pyxtal.call_args_list:
                self.assertLessEqual(sum(call.args[2]), 5,
                    "sample_space_groups tried to call pyxtal with more atoms than it should have."
                )

        mock_pyxtal.reset_mock()

        formulas = Formulas.unary_range("Cu", 10) * Formulas.unary_range("Ag", 10)
        results = list(sample_space_groups(formulas, max_atoms=5))

        with self.subTest("binary"):
            for call in mock_pyxtal.call_args_list:
                self.assertLessEqual(sum(call.args[2]), 5,
                    "sample_space_groups tried to call pyxtal with more atoms than it should have."
                )


if __name__ == "__main__":
    unittest.main()
