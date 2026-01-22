import numpy as np
import pytest
from assyst.crystals import pyxtal, sample_space_groups, Formulas

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
