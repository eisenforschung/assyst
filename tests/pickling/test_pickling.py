import pickle
import numpy as np
from assyst.perturbations import Rattle, Stretch, RandomChoice
from ase import Atoms

def test_pickling_rattle():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)
    r = Rattle(sigma=0.1, rng=42)

    # Progress RNG a bit
    r(at.copy())

    # Pickle and unpickle
    p = pickle.dumps(r)
    r2 = pickle.loads(p)

    # Should produce same next perturbation
    at1 = r(at.copy())
    at2 = r2(at.copy())

    assert np.allclose(at1.positions, at2.positions)

def test_pickling_stretch():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)
    s = Stretch(hydro=0.1, shear=0.1, rng=42)

    # Progress RNG a bit
    s(at.copy())

    # Pickle and unpickle
    p = pickle.dumps(s)
    s2 = pickle.loads(p)

    # Should produce same next perturbation
    at1 = s(at.copy())
    at2 = s2(at.copy())

    assert np.allclose(at1.cell, at2.cell)

def test_pickling_random_choice():
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)
    p1 = Rattle(sigma=0.01, rng=123)
    p2 = Rattle(sigma=0.5, rng=456)
    rc = RandomChoice(p1, p2, chance=0.5, rng=42)

    # Progress RNG a bit
    rc(at.copy())

    # Pickle and unpickle
    p = pickle.dumps(rc)
    rc2 = pickle.loads(p)

    # Should produce same next choices/perturbations
    at1 = rc(at.copy())
    at2 = rc2(at.copy())

    assert np.allclose(at1.positions, at2.positions)
