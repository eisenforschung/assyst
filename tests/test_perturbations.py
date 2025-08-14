import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from ase import Atoms

from assyst.perturbations import rattle, stretch, Rattle, Stretch, Series, RandomChoice, apply_perturbations

class TestPerturbations(unittest.TestCase):

    def setUp(self):
        """Create a simple Atoms object for testing."""
        self.structure = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
        self.single_atom_structure = Atoms('H', positions=[[0, 0, 0]], cell=[10, 10, 10])

    def test_rattle_modifies_positions(self):
        """Test that rattle modifies the positions of the atoms."""
        original_positions = self.structure.get_positions().copy()
        rattled_structure = rattle(self.structure.copy(), sigma=0.1)
        rattled_positions = rattled_structure.get_positions()
        self.assertFalse(np.allclose(original_positions, rattled_positions))

    def test_rattle_raises_value_error_for_single_atom(self):
        """Test that rattle raises a ValueError for a single-atom structure."""
        with self.assertRaises(ValueError):
            rattle(self.single_atom_structure.copy(), sigma=0.1)

    def test_stretch_modifies_cell(self):
        """Test that stretch modifies the cell of the structure."""
        original_cell = self.structure.get_cell().copy()
        stretched_structure = stretch(self.structure.copy(), hydro=0.1, shear=0.1)
        stretched_cell = stretched_structure.get_cell()
        self.assertFalse(np.allclose(original_cell, stretched_cell))

    def test_rattle_class(self):
        """Test the Rattle class."""
        rattle_pert = Rattle(sigma=0.1)
        rattled_structure = rattle_pert(self.structure.copy())
        self.assertIn('rattle(0.1)', rattled_structure.info['perturbation'])

    def test_rattle_class_supercell(self):
        """Test the Rattle class with create_supercells=True."""
        rattle_pert = Rattle(sigma=0.1, create_supercells=True)
        rattled_structure = rattle_pert(self.single_atom_structure.copy())
        self.assertEqual(len(rattled_structure), 8)

    def test_stretch_class(self):
        """Test the Stretch class."""
        stretch_pert = Stretch(hydro=0.1, shear=0.1)
        stretched_structure = stretch_pert(self.structure.copy())
        self.assertIn('stretch(hydro=0.1, shear=0.1)', stretched_structure.info['perturbation'])

    def test_series_class(self):
        """Test the Series class."""
        rattle_pert = Rattle(sigma=0.1)
        stretch_pert = Stretch(hydro=0.1, shear=0.1)
        series_pert = Series((rattle_pert, stretch_pert))
        perturbed_structure = series_pert(self.structure.copy())
        self.assertIn('rattle(0.1)+stretch(hydro=0.1, shear=0.1)', perturbed_structure.info['perturbation'])

    def test_series_addition(self):
        """Test the addition of perturbations."""
        rattle_pert = Rattle(sigma=0.1)
        stretch_pert = Stretch(hydro=0.1, shear=0.1)
        series_pert = rattle_pert + stretch_pert
        perturbed_structure = series_pert(self.structure.copy())
        self.assertIn('rattle(0.1)+stretch(hydro=0.1, shear=0.1)', perturbed_structure.info['perturbation'])

    def test_random_choice_class(self):
        """Test the RandomChoice class."""
        original_rand = np.random.rand

        def side_effect(*args, **kwargs):
            if not args and not kwargs:
                # This is the call from RandomChoice: np.random.rand()
                return side_effect.choice
            # This is a call from rattle or stretch, e.g. np.random.rand(3, 3)
            return original_rand(*args, **kwargs)

        rattle_pert = Rattle(sigma=0.1)
        stretch_pert = Stretch(hydro=0.1, shear=0.1)
        # choice_a is rattle, choice_b is stretch
        random_pert = RandomChoice(rattle_pert, stretch_pert, chance=0.5)

        with patch('numpy.random.rand', side_effect=side_effect):
            # With rand()=0.8, 0.8 > 0.5 is true, so choice_a (rattle) should be called.
            side_effect.choice = 0.8
            perturbed_structure_A = random_pert(self.structure.copy())
            self.assertIn('rattle(0.1)', perturbed_structure_A.info['perturbation'])
            self.assertNotIn('stretch', perturbed_structure_A.info['perturbation'])

            # With rand()=0.2, 0.2 > 0.5 is false, so choice_b (stretch) should be called.
            side_effect.choice = 0.2
            perturbed_structure_B = random_pert(self.structure.copy())
            self.assertIn('stretch(hydro=0.1, shear=0.1)', perturbed_structure_B.info['perturbation'])
            self.assertNotIn('rattle', perturbed_structure_B.info['perturbation'])

    def test_apply_perturbations(self):
        """Test the apply_perturbations function."""
        structures = [self.structure.copy() for _ in range(3)]
        perturbations = [Rattle(sigma=0.1), Stretch(hydro=0.1, shear=0.1)]

        perturbed_structures = list(apply_perturbations(structures, perturbations))

        self.assertEqual(len(perturbed_structures), 6) # 3 structures * 2 perturbations

    def test_apply_perturbations_with_filter(self):
        """Test the apply_perturbations function with a filter."""
        structures = [self.structure.copy()]
        perturbations = [Rattle(sigma=0.1)]

        # This filter should always return False
        false_filter = lambda s: False

        perturbed_structures = list(apply_perturbations(structures, perturbations, filters=false_filter))
        self.assertEqual(len(perturbed_structures), 0)

    def test_apply_perturbations_value_error(self):
        """Test that apply_perturbations handles ValueError."""
        structures = [self.single_atom_structure.copy()]
        perturbations = [Rattle(sigma=0.1, create_supercells=False)] # This will raise ValueError

        # The ValueError from Rattle should be caught, and no structures should be yielded.
        perturbed_structures = list(apply_perturbations(structures, perturbations))
        self.assertEqual(len(perturbed_structures), 0)

    def test_stretch_strain_distribution(self):
        """Test that stretch applies strain within the correct ranges."""
        for _ in range(10): # Repeat test 10 times with different random values
            hydro = np.random.uniform(0.05, 0.3)
            shear = np.random.uniform(0.05, 0.3)
            minimum_strain = np.random.uniform(1e-4, 1e-2)

            # Ensure hydro and shear are greater than minimum_strain
            hydro = max(hydro, minimum_strain + 1e-3)
            shear = max(shear, minimum_strain + 1e-3)

            original_cell = self.structure.get_cell().copy()
            stretched_structure = stretch(self.structure.copy(), hydro=hydro, shear=shear, minimum_strain=minimum_strain)
            stretched_cell = stretched_structure.get_cell()

            strain_modifier = np.linalg.inv(original_cell) @ stretched_cell

            epsilon = strain_modifier - np.identity(3)

            self.assertTrue(np.allclose(epsilon, epsilon.T), "Strain matrix should be symmetric")

            # Check diagonal elements
            diag_strains = np.diag(epsilon)
            for strain_val in diag_strains:
                self.assertTrue(minimum_strain <= abs(strain_val) <= hydro)

            # Check off-diagonal elements
            off_diag_indices = np.triu_indices(3, k=1)
            off_diag_strains = epsilon[off_diag_indices]
            for strain_val in off_diag_strains:
                self.assertTrue(minimum_strain <= abs(strain_val) <= shear)


    def test_perturbation_info_concatenation(self):
        """Test that perturbation info is concatenated."""
        rattle_pert = Rattle(sigma=0.1)
        stretch_pert = Stretch(hydro=0.1, shear=0.1)

        structure = self.structure.copy()
        structure = rattle_pert(structure)
        structure = stretch_pert(structure)

        self.assertIn('rattle(0.1)+stretch(hydro=0.1, shear=0.1)', structure.info['perturbation'])

if __name__ == '__main__':
    unittest.main()
