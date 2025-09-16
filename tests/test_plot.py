import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ase import Atoms
from assyst.plot import _volume, _energy, _concentration, volume_histogram, size_histogram, concentration_histogram, distance_histogram, energy_volume

try:
    import matscipy
except ImportError:
    matscipy = None


class TestPlotHelpers(unittest.TestCase):
    def setUp(self):
        self.s1 = Atoms('H2', positions=[[0,0,0], [1,0,0]], cell=[10,10,10])
        self.s2 = Atoms('O', positions=[[0,0,0]], cell=[5,5,5])
        self.s1.calc = MagicMock()
        self.s1.calc.get_potential_energy.return_value = -2.0
        self.s2.calc = MagicMock()
        self.s2.calc.get_potential_energy.return_value = -5.0
        self.structures = [self.s1, self.s2]

    def test_volume(self):
        volumes = _volume(self.structures)
        self.assertAlmostEqual(volumes[0], 500.0)
        self.assertAlmostEqual(volumes[1], 125.0)

    def test_energy(self):
        energies = _energy(self.structures)
        self.assertAlmostEqual(energies[0], -1.0)
        self.assertAlmostEqual(energies[1], -5.0)

    def test_concentration(self):
        concentrations = _concentration(self.structures)
        self.assertTrue('H' in concentrations)
        self.assertTrue('O' in concentrations)
        np.testing.assert_array_almost_equal(concentrations['H'], [1.0, 0.0])
        np.testing.assert_array_almost_equal(concentrations['O'], [0.0, 1.0])

    def test_concentration_with_elements(self):
        concentrations = _concentration(self.structures, elements=['H'])
        self.assertTrue('H' in concentrations)
        self.assertFalse('O' in concentrations)

class TestPlotFunctions(unittest.TestCase):
    def setUp(self):
        self.structures = [Atoms('H2O', positions=[[0,0,0], [1,0,0], [0,1,0]], cell=[10,10,10])]

    @patch('matplotlib.pyplot.hist')
    def test_volume_histogram(self, mock_hist):
        volume_histogram(self.structures)
        mock_hist.assert_called_once()

    @patch('matplotlib.pyplot.hist')
    def test_size_histogram(self, mock_hist):
        size_histogram(self.structures)
        mock_hist.assert_called_once()

    @patch('matplotlib.pyplot.bar')
    def test_concentration_histogram(self, mock_bar):
        concentration_histogram(self.structures)
        mock_bar.assert_called()

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    @patch('matplotlib.pyplot.hist')
    def test_distance_histogram(self, mock_hist):
        distance_histogram(self.structures)
        mock_hist.assert_called_once()

    @patch('matplotlib.pyplot.scatter')
    def test_energy_volume_scatter(self, mock_scatter):
        s = Atoms('H', cell=[10,10,10])
        s.calc = MagicMock()
        s.calc.get_potential_energy.return_value = -1.0
        energy_volume([s])
        mock_scatter.assert_called_once()

    @patch('matplotlib.pyplot.hexbin')
    def test_energy_volume_hexbin(self, mock_hexbin):
        s = Atoms('H', cell=[10,10,10])
        s.calc = MagicMock()
        s.calc.get_potential_energy.return_value = -1.0
        energy_volume([s] * 1001)
        mock_hexbin.assert_called_once()
