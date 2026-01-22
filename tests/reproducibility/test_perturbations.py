import numpy as np
import pytest
from assyst.perturbations import rattle, element_scaled_rattle, stretch, Rattle, ElementScaledRattle, Stretch, RandomChoice
from ase import Atoms

def test_rattle_reproducibility():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)

    at1 = at.copy()
    rattle(at1, sigma=0.1, rng=42)

    at2 = at.copy()
    rattle(at2, sigma=0.1, rng=42)

    assert np.allclose(at1.positions, at2.positions)

def test_element_scaled_rattle_reproducibility():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)
    reference = {"Al": 1.0}

    at1 = at.copy()
    element_scaled_rattle(at1, sigma=0.1, reference=reference, rng=42)

    at2 = at.copy()
    element_scaled_rattle(at2, sigma=0.1, reference=reference, rng=42)

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

    # ElementScaledRattle class
    esr1 = ElementScaledRattle(sigma=0.1, reference={"Al": 1.0}, rng=42)
    esr2 = ElementScaledRattle(sigma=0.1, reference={"Al": 1.0}, rng=42)
    at1 = esr1(at.copy())
    at2 = esr2(at.copy())
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
