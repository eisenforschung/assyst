import numpy as np
import pytest
from assyst.crystals import pyxtal, sample_space_groups, Formulas
from assyst.perturbations import rattle, stretch, Rattle, Stretch, RandomChoice
from ase import Atoms

def test_pyxtal_reproducibility():
    species = ("Al",)
    num_ions = (4,)
    group = 225

    # Same seed should produce same structures
    s1 = pyxtal(group, species, num_ions, rng=42)
    s2 = pyxtal(group, species, num_ions, rng=42)
    assert np.allclose(s1.positions, s2.positions)

    # Different seed should produce different structures (most likely)
    s3 = pyxtal(group, species, num_ions, rng=43)
    assert not np.allclose(s1.positions, s3.positions)

def test_pyxtal_generator_progression():
    species = ("Al",)
    num_ions = (4,)
    group = 225
    rng = np.random.default_rng(42)

    # Passing the same generator should produce different structures in subsequent calls
    s1 = pyxtal(group, species, num_ions, rng=rng)
    s2 = pyxtal(group, species, num_ions, rng=rng)
    assert not np.allclose(s1.positions, s2.positions)

def test_sample_space_groups_reproducibility():
    formulas = Formulas(({'Al': 4},))
    spacegroups = [225]

    at1 = list(sample_space_groups(formulas, spacegroups, rng=42))[0]
    at2 = list(sample_space_groups(formulas, spacegroups, rng=42))[0]
    assert np.allclose(at1.positions, at2.positions)

def test_rattle_reproducibility():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)

    at1 = at.copy()
    rattle(at1, sigma=0.1, rng=42)

    at2 = at.copy()
    rattle(at2, sigma=0.1, rng=42)

    assert np.allclose(at1.positions, at2.positions)

def test_stretch_reproducibility():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)

    at1 = at.copy()
    stretch(at1, hydro=0.1, shear=0.1, rng=42)

    at2 = at.copy()
    stretch(at2, hydro=0.1, shear=0.1, rng=42)

    assert np.allclose(at1.cell, at2.cell)

def test_perturbation_classes_reproducibility():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)

    # Two different instances with SAME seed should produce SAME results on THEIR FIRST calls
    r1 = Rattle(sigma=0.1, rng=42)
    r2 = Rattle(sigma=0.1, rng=42)
    at1 = r1(at.copy())
    at2 = r2(at.copy())
    assert np.allclose(at1.positions, at2.positions)

    # Stretch class
    s1 = Stretch(hydro=0.1, shear=0.1, rng=42)
    s2 = Stretch(hydro=0.1, shear=0.1, rng=42)
    at1 = s1(at.copy())
    at2 = s2(at.copy())
    assert np.allclose(at1.cell, at2.cell)

def test_random_choice_reproducibility():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)

    # Define two distinct perturbations
    # With same seed, should pick the same one AND the picked one should have same result
    rc1 = RandomChoice(Rattle(sigma=0.01, rng=123), Rattle(sigma=0.5, rng=123), chance=0.5, rng=42)
    rc2 = RandomChoice(Rattle(sigma=0.01, rng=123), Rattle(sigma=0.5, rng=123), chance=0.5, rng=42)

    at1 = rc1(at.copy())
    at2 = rc2(at.copy())

    assert np.allclose(at1.positions, at2.positions)

def test_perturbation_progression():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)

    # Rattle class with seed should produce DIFFERENT results on subsequent calls
    r = Rattle(sigma=0.1, rng=42)
    at1 = r(at.copy())
    at2 = r(at.copy())
    assert not np.allclose(at1.positions, at2.positions)

    # But two DIFFERENT instances with same seed should produce SAME results for their FIRST call
    r1 = Rattle(sigma=0.1, rng=42)
    r2 = Rattle(sigma=0.1, rng=42)
    at_r1 = r1(at.copy())
    at_r2 = r2(at.copy())
    assert np.allclose(at_r1.positions, at_r2.positions)

def test_random_choice_progression():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)
    p1 = Rattle(sigma=0.01, rng=123)
    p2 = Rattle(sigma=0.5, rng=456)

    rc = RandomChoice(p1, p2, chance=0.5, rng=42)

    # Collect choices over many calls
    results = []
    for _ in range(100):
        res = rc(at.copy())
        # Check if it was p1 or p2 by looking at max displacement
        disp = np.linalg.norm(res.positions - at.positions, axis=1).max()
        results.append(disp > 0.1) # True if p2 was chosen

    # It shouldn't be all the same choice
    assert any(results)
    assert not all(results)
