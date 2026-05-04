import unittest
from unittest.mock import patch
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter
from assyst.relaxations import Relax, CellRelax, VolumeRelax, SymmetryRelax, FullRelax, relax
from assyst.calculators import AseCalculatorConfig, Morse
import numpy as np


class MockCalculator:
    def get_potential_energy(self, atoms=None):
        return 0.0
    def get_forces(self, atoms=None):
        return [[0.0, 0.0, 0.0]]
    def get_stress(self, atoms=None):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestRelax(unittest.TestCase):
    def setUp(self):
        self.structure = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
        self.structure.calc = MockCalculator()

    @patch('ase.optimize.LBFGS.run')
    def test_relax_runs(self, mock_run):
        relaxer = Relax()
        relaxed_structure = relaxer.relax(self.structure)
        mock_run.assert_called_once()
        self.assertIsInstance(relaxed_structure.calc, SinglePointCalculator)

    @patch('ase.optimize.BFGS.run')
    def test_relax_bfgs(self, mock_run):
        relaxer = Relax(algorithm="BFGS")
        relaxed_structure = relaxer.relax(self.structure)
        mock_run.assert_called_once()
        self.assertIsInstance(relaxed_structure.calc, SinglePointCalculator)

    @patch('ase.optimize.FIRE.run')
    def test_relax_fire(self, mock_run):
        relaxer = Relax(algorithm="FIRE")
        relaxed_structure = relaxer.relax(self.structure)
        mock_run.assert_called_once()
        self.assertIsInstance(relaxed_structure.calc, SinglePointCalculator)

    @patch('ase.optimize.LBFGS.run')
    def test_cell_relax(self, mock_run):
        relaxer = CellRelax()
        relaxer.relax(self.structure)
        mock_run.assert_called_once()

    @patch('ase.optimize.LBFGS.run')
    def test_volume_relax(self, mock_run):
        relaxer = VolumeRelax(pressure=1.0)
        relaxer.relax(self.structure)
        mock_run.assert_called_once()

    @patch('ase.optimize.LBFGS.run')
    def test_symmetry_relax(self, mock_run):
        relaxer = SymmetryRelax()
        relaxer.relax(self.structure)
        mock_run.assert_called_once()

    @patch('ase.optimize.LBFGS.run')
    def test_full_relax(self, mock_run):
        relaxer = FullRelax(pressure=1.0)
        relaxer.relax(self.structure)
        mock_run.assert_called_once()

    @patch('assyst.relaxations.Relax.relax')
    def test_relax_function_with_calc_object(self, mock_relax_method):
        settings = Relax()
        calculator = MockCalculator()
        structures = [self.structure]

        list(relax(structures, settings, calculator))

        self.assertIs(mock_relax_method.call_args[0][0].calc, calculator)

    @patch('assyst.relaxations.Relax.relax')
    def test_relax_function_with_calc_config(self, mock_relax_method):
        class MockCalcConfig(AseCalculatorConfig):
            def get_calculator(self):
                return MockCalculator()

        settings = Relax()
        calculator_config = MockCalcConfig()
        structures = [self.structure]

        list(relax(structures, settings, calculator_config))

        self.assertIsInstance(mock_relax_method.call_args[0][0].calc, MockCalculator)


class TestRelaxFiltersAndConstraints(unittest.TestCase):
    """Verify apply_filter_and_constraints returns the correct filter/constraint types."""

    def _make_structure(self):
        s = bulk('Cu', cubic=True)
        s.calc = Morse().get_calculator()
        return s

    def test_base_relax_returns_structure_unchanged(self):
        s = self._make_structure()
        r = Relax()
        result = r.apply_filter_and_constraints(s)
        self.assertIs(result, s)
        self.assertEqual(len(s.constraints), 0)

    def test_cell_relax_returns_frechet_cell_filter(self):
        s = self._make_structure()
        r = CellRelax()
        result = r.apply_filter_and_constraints(s)
        self.assertIsInstance(result, FrechetCellFilter)
        self.assertTrue(result.constant_volume)

    def test_cell_relax_adds_fix_atoms_constraint(self):
        s = self._make_structure()
        CellRelax().apply_filter_and_constraints(s)
        self.assertEqual(len(s.constraints), 1)
        self.assertIsInstance(s.constraints[0], FixAtoms)

    def test_volume_relax_returns_frechet_cell_filter_with_hydrostatic(self):
        s = self._make_structure()
        r = VolumeRelax(pressure=2.0)
        result = r.apply_filter_and_constraints(s)
        self.assertIsInstance(result, FrechetCellFilter)
        self.assertTrue(result.hydrostatic_strain)
        self.assertEqual(result.scalar_pressure, 2.0)

    def test_volume_relax_adds_fix_atoms_constraint(self):
        s = self._make_structure()
        VolumeRelax().apply_filter_and_constraints(s)
        self.assertEqual(len(s.constraints), 1)
        self.assertIsInstance(s.constraints[0], FixAtoms)

    def test_volume_relax_default_pressure_is_zero(self):
        s = self._make_structure()
        result = VolumeRelax().apply_filter_and_constraints(s)
        self.assertEqual(result.scalar_pressure, 0.0)

    def test_symmetry_relax_returns_frechet_cell_filter(self):
        s = self._make_structure()
        r = SymmetryRelax(pressure=0.5)
        result = r.apply_filter_and_constraints(s)
        self.assertIsInstance(result, FrechetCellFilter)
        self.assertEqual(result.scalar_pressure, 0.5)

    def test_symmetry_relax_adds_fix_symmetry_constraint(self):
        s = self._make_structure()
        SymmetryRelax().apply_filter_and_constraints(s)
        self.assertEqual(len(s.constraints), 1)
        self.assertIsInstance(s.constraints[0], FixSymmetry)

    def test_full_relax_returns_frechet_cell_filter(self):
        s = self._make_structure()
        r = FullRelax(pressure=1.5)
        result = r.apply_filter_and_constraints(s)
        self.assertIsInstance(result, FrechetCellFilter)
        self.assertEqual(result.scalar_pressure, 1.5)

    def test_full_relax_does_not_add_constraints(self):
        s = self._make_structure()
        FullRelax().apply_filter_and_constraints(s)
        self.assertEqual(len(s.constraints), 0)


class TestRelaxWithRealCalculator(unittest.TestCase):
    """Integration tests using the Morse potential to verify real relaxation behaviour."""

    def _make_structure(self):
        s = bulk('Cu', cubic=True)
        s.calc = Morse().get_calculator()
        return s

    def test_relax_returns_single_point_calculator_with_valid_energy(self):
        s = self._make_structure()
        result = Relax(max_steps=50).relax(s)
        self.assertIsInstance(result.calc, SinglePointCalculator)
        energy = result.calc.get_potential_energy()
        self.assertIsInstance(energy, float)
        self.assertFalse(np.isnan(energy))

    def test_relax_returns_forces_with_correct_shape(self):
        s = self._make_structure()
        result = Relax(max_steps=50).relax(s)
        forces = result.calc.get_forces()
        self.assertEqual(forces.shape, (len(s), 3))

    def test_relax_returns_stress_with_correct_shape(self):
        s = self._make_structure()
        result = Relax(max_steps=50).relax(s)
        stress = result.calc.get_stress()
        self.assertEqual(stress.shape, (6,))

    def test_relax_clears_constraints_on_output(self):
        s = self._make_structure()
        result = CellRelax(max_steps=10).relax(s)
        self.assertEqual(len(result.constraints), 0)

    def test_relax_does_not_modify_input_structure_positions(self):
        s = self._make_structure()
        original_positions = s.get_positions().copy()
        Relax(max_steps=5).relax(s)
        np.testing.assert_array_equal(s.get_positions(), original_positions)

    def test_relax_assigns_new_uuid(self):
        s = self._make_structure()
        s.info['uuid'] = 'original-uuid'
        result = Relax(max_steps=5).relax(s)
        self.assertIn('uuid', result.info)
        self.assertNotEqual(result.info['uuid'], 'original-uuid')

    def test_relax_tracks_uuid_lineage(self):
        s = self._make_structure()
        s.info['uuid'] = 'parent-uuid'
        result = Relax(max_steps=5).relax(s)
        self.assertIn('lineage', result.info)
        self.assertIn('parent-uuid', result.info['lineage'])

    def test_relax_preserves_seed(self):
        s = self._make_structure()
        s.info['uuid'] = 'first-uuid'
        s.info['seed'] = 'seed-value'
        result = Relax(max_steps=5).relax(s)
        self.assertEqual(result.info['seed'], 'seed-value')

    def test_relax_assigns_seed_if_absent(self):
        s = self._make_structure()
        result = Relax(max_steps=5).relax(s)
        self.assertIn('seed', result.info)
        self.assertEqual(result.info['seed'], result.info['uuid'])

    def test_relax_reduces_energy(self):
        s = bulk('Cu', cubic=True)
        s.positions[0] += 0.3
        s.calc = Morse().get_calculator()
        initial_energy = s.get_potential_energy()
        result = Relax(max_steps=100).relax(s)
        self.assertLess(result.calc.get_potential_energy(), initial_energy)

    def test_full_relax_converges(self):
        s = self._make_structure()
        result = FullRelax(max_steps=200).relax(s)
        forces = result.calc.get_forces()
        self.assertTrue(np.all(np.abs(forces) < 1e-2))


class TestRelaxDataclassEquality(unittest.TestCase):
    """Test frozen dataclass equality and hash semantics."""

    def test_relax_equal_with_same_params(self):
        self.assertEqual(Relax(max_steps=50), Relax(max_steps=50))

    def test_relax_not_equal_with_different_params(self):
        self.assertNotEqual(Relax(max_steps=50), Relax(max_steps=100))

    def test_relax_hashable(self):
        r = Relax()
        self.assertEqual(hash(r), hash(r))
        s = {r}
        self.assertIn(r, s)

    def test_cell_relax_equal_with_same_params(self):
        self.assertEqual(CellRelax(max_steps=50), CellRelax(max_steps=50))

    def test_volume_relax_equal_with_same_pressure(self):
        self.assertEqual(VolumeRelax(pressure=1.0), VolumeRelax(pressure=1.0))

    def test_volume_relax_not_equal_with_different_pressure(self):
        self.assertNotEqual(VolumeRelax(pressure=0.0), VolumeRelax(pressure=1.0))

    def test_symmetry_relax_equal_with_same_params(self):
        self.assertEqual(SymmetryRelax(pressure=0.5), SymmetryRelax(pressure=0.5))

    def test_full_relax_equal_with_same_params(self):
        self.assertEqual(FullRelax(pressure=0.0), FullRelax(pressure=0.0))

    def test_different_relax_types_not_equal(self):
        self.assertNotEqual(Relax(), CellRelax())
        self.assertNotEqual(CellRelax(), VolumeRelax())
        self.assertNotEqual(SymmetryRelax(), FullRelax())

    def test_relax_default_values(self):
        r = Relax()
        self.assertEqual(r.max_steps, 100)
        self.assertAlmostEqual(r.force_tolerance, 1e-3)
        self.assertEqual(r.algorithm, "LBFGS")

    def test_volume_relax_default_pressure(self):
        self.assertEqual(VolumeRelax().pressure, 0.0)

    def test_symmetry_relax_default_pressure(self):
        self.assertEqual(SymmetryRelax().pressure, 0.0)

    def test_full_relax_default_pressure(self):
        self.assertEqual(FullRelax().pressure, 0.0)


class TestRelaxFunction(unittest.TestCase):
    """Tests for the module-level relax() generator function."""

    def setUp(self):
        self.structures = [bulk('Cu', cubic=True) for _ in range(3)]
        for s in self.structures:
            s.positions[0] += np.random.uniform(-0.1, 0.1, size=3)

    def test_relax_function_yields_correct_number_of_structures(self):
        results = list(relax(self.structures, Relax(max_steps=5), Morse()))
        self.assertEqual(len(results), len(self.structures))

    def test_relax_function_does_not_mutate_input(self):
        original_positions = [s.get_positions().copy() for s in self.structures]
        list(relax(self.structures, Relax(max_steps=5), Morse()))
        for s, orig in zip(self.structures, original_positions):
            np.testing.assert_array_equal(s.get_positions(), orig)

    def test_relax_function_attaches_single_point_calculator(self):
        results = list(relax(self.structures, Relax(max_steps=5), Morse()))
        for result in results:
            self.assertIsInstance(result.calc, SinglePointCalculator)

    def test_relax_function_with_ase_calculator_config(self):
        results = list(relax(self.structures, Relax(max_steps=5), Morse()))
        self.assertEqual(len(results), 3)

    def test_relax_function_with_raw_ase_calculator(self):
        calc = Morse().get_calculator()
        results = list(relax(self.structures, Relax(max_steps=5), calc))
        self.assertEqual(len(results), 3)

    def test_relax_function_is_lazy_generator(self):
        call_count = [0]
        original_relax = Relax.relax

        def counting_relax(self_inner, structure):
            call_count[0] += 1
            return original_relax(self_inner, structure)

        with patch.object(Relax, 'relax', counting_relax):
            gen = relax(self.structures, Relax(max_steps=5), Morse())
            self.assertEqual(call_count[0], 0, "relax should not be called before iteration")
            next(gen)
            self.assertEqual(call_count[0], 1)


if __name__ == '__main__':
    unittest.main()
