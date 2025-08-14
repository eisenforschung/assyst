import unittest
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from assyst.filters import EnergyFilter, ForceFilter, CalculatorFilter

class MockCalculatorFilter(CalculatorFilter):
    def __call__(self, structure: Atoms) -> bool:
        return self._check(structure)

class TestCalculatorFilters(unittest.TestCase):
    def test_calculator_filter_check(self):
        filter = MockCalculatorFilter(missing='error')

        # No calculator
        structure = Atoms('Cu')
        with self.assertRaises(ValueError):
            filter._check(structure)

        # Wrong calculator
        structure.calc = "dummy"
        with self.assertRaises(ValueError):
            filter._check(structure)

        # Correct calculator
        structure.calc = SinglePointCalculator(structure, energy=0.0, forces=np.zeros((1, 3)))
        self.assertTrue(filter._check(structure))

        # Missing ignore
        filter = MockCalculatorFilter(missing='ignore')
        structure = Atoms('Cu')
        self.assertFalse(filter(structure))

    def test_energy_filter(self):
        # Test with no calculator
        filter = EnergyFilter(max_energy=1.0, missing='ignore')
        structure = Atoms('Cu')
        self.assertTrue(filter(structure)) # Should pass if _check is false

        filter = EnergyFilter(max_energy=1.0)
        structure = Atoms('Cu')
        structure.calc = SinglePointCalculator(structure, energy=2.0)
        self.assertFalse(filter(structure)) # energy 2.0 > 1.0

        structure.calc = SinglePointCalculator(structure, energy=0.5)
        self.assertTrue(filter(structure)) # energy 0.5 < 1.0

        filter = EnergyFilter(min_energy=0.0, max_energy=1.0)
        structure.calc = SinglePointCalculator(structure, energy=-0.5)
        self.assertFalse(filter(structure)) # energy -0.5 < 0.0

    def test_force_filter(self):
        # Test with no calculator
        filter = ForceFilter(max_force=1.0, missing='ignore')
        structure = Atoms('Cu')
        self.assertTrue(filter(structure)) # Should pass if _check is false

        filter = ForceFilter(max_force=1.0)
        structure = Atoms('Cu')
        forces = np.array([[1.1, 0.0, 0.0]])
        structure.calc = SinglePointCalculator(structure, forces=forces)
        self.assertFalse(filter(structure)) # force 1.1 > 1.0

        forces = np.array([[0.5, 0.0, 0.0]])
        structure.calc = SinglePointCalculator(structure, forces=forces)
        self.assertTrue(filter(structure)) # force 0.5 < 1.0
