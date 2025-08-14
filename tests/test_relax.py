import unittest
from unittest.mock import patch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from assyst.relax import Relax, CellRelax, VolumeRelax, SymmetryRelax, FullRelax, relax
from assyst.calculators import AseCalculatorConfig

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

    @patch('assyst.relax.Relax.relax')
    def test_relax_function_with_calc_object(self, mock_relax_method):
        settings = Relax()
        calculator = MockCalculator()
        structures = [self.structure]

        list(relax(settings, calculator, structures))

        self.assertIs(mock_relax_method.call_args[0][0].calc, calculator)

    @patch('assyst.relax.Relax.relax')
    def test_relax_function_with_calc_config(self, mock_relax_method):
        class MockCalcConfig(AseCalculatorConfig):
            def get_calculator(self):
                return MockCalculator()

        settings = Relax()
        calculator_config = MockCalcConfig()
        structures = [self.structure]

        list(relax(settings, calculator_config, structures))

        self.assertIsInstance(mock_relax_method.call_args[0][0].calc, MockCalculator)
