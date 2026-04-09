import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ase import Atoms
from assyst.plot import (
    _volume,
    _energy,
    _concentration,
    _lattice_parameters,
    _lattice_angles,
    _aspect_ratio,
    _reduce_distances,
    _distance_xlabel,
    volume_histogram,
    size_histogram,
    concentration_histogram,
    distance_histogram,
    radial_distribution,
    energy_histogram,
    energy_distance,
    energy_volume,
    lattice_parameter_histogram,
    lattice_angle_histogram,
    aspect_ratio_histogram,
)

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

class TestReduceDistances(unittest.TestCase):
    @unittest.skipIf(matscipy is None, "matscipy not installed")
    def test_reduce_min(self):
        s = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
        result = _reduce_distances([s], rmax=6.0, reduce="min")
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 1.0)

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    def test_reduce_skips_no_neighbors(self):
        no_neighbors = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10])
        s = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
        result = _reduce_distances([no_neighbors, s], rmax=6.0, reduce="min")
        self.assertEqual(len(result), 1)

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    def test_reduce_none_concatenates(self):
        s = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
        result = _reduce_distances([s], rmax=6.0, reduce=None)
        self.assertGreater(len(result), 0)

    def test_distance_xlabel_presets(self):
        self.assertIn("Minimum", _distance_xlabel("min"))
        self.assertIn("Mean", _distance_xlabel("mean"))
        self.assertIn("Distance", _distance_xlabel(None))
        self.assertIn("Distance", _distance_xlabel(lambda x: x[0]))


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
        distance_histogram(self.structures, reduce="min")
        mock_hist.assert_called_once()

        distance_histogram(self.structures, reduce="mean")
        distance_histogram(self.structures, reduce=None)

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    @patch('matplotlib.pyplot.hist')
    def test_distance_histogram_no_neighbors(self, mock_hist):
        # Single-atom structure has no neighbors; should not raise
        no_neighbors = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10])
        distance_histogram([no_neighbors] + self.structures, reduce="min")
        distance_histogram([no_neighbors] + self.structures, reduce="mean")

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    @patch('matplotlib.pyplot.hist')
    def test_radial_distribution(self, mock_hist):
        radial_distribution(self.structures)
        mock_hist.assert_called_once()

    @patch('matplotlib.pyplot.hist')
    def test_energy_histogram(self, mock_hist):
        s = Atoms('H', cell=[10,10,10])
        s.calc = MagicMock()
        s.calc.get_potential_energy.return_value = -1.0
        energy_histogram([s])
        mock_hist.assert_called_once()

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    @patch('matplotlib.pyplot.scatter')
    def test_energy_distance_scatter(self, mock_scatter):
        s = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
        s.calc = MagicMock()
        s.calc.get_potential_energy.return_value = -2.0
        energy_distance([s], reduce="min")
        mock_scatter.assert_called_once()

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    @patch('matplotlib.pyplot.scatter')
    def test_energy_distance_mean(self, mock_scatter):
        s = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
        s.calc = MagicMock()
        s.calc.get_potential_energy.return_value = -2.0
        energy_distance([s], reduce="mean")
        mock_scatter.assert_called_once()

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    @patch('matplotlib.pyplot.hexbin')
    def test_energy_distance_hexbin(self, mock_hexbin):
        s = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
        s.calc = MagicMock()
        s.calc.get_potential_energy.return_value = -2.0
        energy_distance([s] * 1001, reduce="min")
        mock_hexbin.assert_called_once()

    @unittest.skipIf(matscipy is None, "matscipy not installed")
    @patch('matplotlib.pyplot.scatter')
    def test_energy_distance_no_neighbors(self, mock_scatter):
        # Structures with no neighbors should be silently skipped
        no_neighbors = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10])
        no_neighbors.calc = MagicMock()
        no_neighbors.calc.get_potential_energy.return_value = -1.0
        s = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]], cell=[10, 10, 10])
        s.calc = MagicMock()
        s.calc.get_potential_energy.return_value = -2.0
        energy_distance([no_neighbors, s], reduce="min")
        mock_scatter.assert_called_once()

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


class TestCellShapeHelpers(unittest.TestCase):
    def setUp(self):
        self.s1 = Atoms('Cu', cell=[3.0, 4.0, 5.0], pbc=True)
        self.s2 = Atoms('Fe', cell=[2.5, 2.5, 6.0], pbc=True)
        self.structures = [self.s1, self.s2]

    def test_lattice_parameters(self):
        params = _lattice_parameters(self.structures)
        self.assertIn('a', params)
        self.assertIn('b', params)
        self.assertIn('c', params)
        np.testing.assert_array_almost_equal(params['a'], [3.0, 2.5])
        np.testing.assert_array_almost_equal(params['b'], [4.0, 2.5])
        np.testing.assert_array_almost_equal(params['c'], [5.0, 6.0])

    def test_lattice_angles(self):
        angles = _lattice_angles(self.structures)
        self.assertEqual(len(angles), 3)
        # Orthogonal cells have all angles = 90
        for values in angles.values():
            np.testing.assert_array_almost_equal(values, [90.0, 90.0])

    def test_aspect_ratio(self):
        ratios = _aspect_ratio(self.structures)
        self.assertAlmostEqual(ratios[0], 5.0 / 3.0)
        self.assertAlmostEqual(ratios[1], 6.0 / 2.5)


class TestCellShapePlots(unittest.TestCase):
    def setUp(self):
        self.structures = [
            Atoms('Cu', cell=[3.0, 4.0, 5.0], pbc=True),
            Atoms('Fe', cell=[2.5, 3.5, 6.0], pbc=True),
        ]

    @patch('seaborn.histplot')
    @patch('matplotlib.pyplot.legend')
    def test_lattice_parameter_histogram(self, mock_legend, mock_histplot):
        lattice_parameter_histogram(self.structures)
        self.assertEqual(mock_histplot.call_count, 3)
        mock_legend.assert_called_once()

    @patch('seaborn.histplot')
    @patch('matplotlib.pyplot.legend')
    def test_lattice_angle_histogram(self, mock_legend, mock_histplot):
        lattice_angle_histogram(self.structures)
        self.assertEqual(mock_histplot.call_count, 3)
        mock_legend.assert_called_once()

    @patch('matplotlib.pyplot.hist')
    def test_aspect_ratio_histogram(self, mock_hist):
        aspect_ratio_histogram(self.structures)
        mock_hist.assert_called_once()
