import pickle
import numpy as np
from hypothesis import given, strategies as st
from assyst.perturbations import Rattle, Stretch, RandomChoice
from ase import Atoms

@given(st.floats(min_value=0.01, max_value=1.0), st.integers(min_value=0, max_value=1000), st.integers(min_value=1, max_value=10))
def test_pickling_rattle(sigma, seed, n_steps):
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)
    r = Rattle(sigma=sigma, rng=seed)

    # Progress RNG
    for _ in range(n_steps):
        r(at.copy())

    # Pickle and unpickle
    p = pickle.dumps(r)
    r2 = pickle.loads(p)

    # Should produce same next perturbation
    at1 = r(at.copy())
    at2 = r2(at.copy())

    assert np.allclose(at1.positions, at2.positions)

@given(st.floats(min_value=0.01, max_value=0.5), st.floats(min_value=0.01, max_value=0.5), st.integers(min_value=0, max_value=1000), st.integers(min_value=1, max_value=10))
def test_pickling_stretch(hydro, shear, seed, n_steps):
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)
    s = Stretch(hydro=hydro, shear=shear, rng=seed)

    # Progress RNG
    for _ in range(n_steps):
        s(at.copy())

    # Pickle and unpickle
    p = pickle.dumps(s)
    s2 = pickle.loads(p)

    # Should produce same next perturbation
    at1 = s(at.copy())
    at2 = s2(at.copy())

    assert np.allclose(at1.cell, at2.cell)

@given(st.floats(min_value=0.0, max_value=1.0), st.integers(min_value=0, max_value=1000), st.integers(min_value=1, max_value=10))
def test_pickling_random_choice(chance, seed, n_steps):
    at = Atoms('Al4', positions=[[0,0,0], [1,1,1], [2,2,2], [3,3,3]], cell=[4,4,4], pbc=True)
    p1 = Rattle(sigma=0.01, rng=seed)
    p2 = Rattle(sigma=0.5, rng=seed+1)
    rc = RandomChoice(p1, p2, chance=chance, rng=seed+2)

    # Progress RNG
    for _ in range(n_steps):
        rc(at.copy())

    # Pickle and unpickle
    p = pickle.dumps(rc)
    rc2 = pickle.loads(p)

    # Should produce same next choices/perturbations
    at1 = rc(at.copy())
    at2 = rc2(at.copy())

    assert np.allclose(at1.positions, at2.positions)
