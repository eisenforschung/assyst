import unittest
from assyst.calculators import Morse, Grace
from ase.calculators.morse import MorsePotential
from ase.calculators.calculator import Calculator

try:
    import tensorpotential
except ImportError:
    tensorpotential = None


class TestCalculators(unittest.TestCase):
    def test_morse(self):
        config = Morse(epsilon=2.0, r0=2.0, rho0=2.0)
        calc = config.get_calculator()
        self.assertIsInstance(calc, MorsePotential)
        self.assertEqual(calc.parameters['epsilon'], 2.0)
        self.assertEqual(calc.parameters['r0'], 2.0)
        self.assertEqual(calc.parameters['rho0'], 2.0)

    @unittest.skipIf(tensorpotential is None, "tensorpotential not installed")
    def test_grace(self):
        config = Grace()
        calc = config.get_calculator()
        self.assertIsInstance(calc, Calculator)
