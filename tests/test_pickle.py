import pickle
import pytest
import numpy as np
import dataclasses
import types
from ase import Atoms
import assyst.perturbations as perturbations
import assyst.filters as filters
import assyst.calculators as calculators
import assyst.crystals as crystals
import assyst.neighbors as neighbors
import assyst.plot as plot
import assyst.relax as relax

try:
    import tensorpotential
    HAS_TENSORPOTENTIAL = True
except ImportError:
    HAS_TENSORPOTENTIAL = False

def assert_equal_recursive(a, b):
    if type(a) is not type(b):
        return False

    if isinstance(a, (int, float, str, bool, type(None))):
        return a == b

    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(assert_equal_recursive(ai, bi) for ai, bi in zip(a, b))

    if isinstance(a, dict):
        if len(a) != len(b):
            return False
        return all(k in b and assert_equal_recursive(a[k], b[k]) for k in a)

    if dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            if field.name == 'rng':
                continue
            if not assert_equal_recursive(getattr(a, field.name), getattr(b, field.name)):
                return False
        return True

    if isinstance(a, np.random.Generator):
        return True # We assume Generators are correctly pickled/unpickled

    return a == b

@pytest.mark.parametrize("obj", [
    perturbations.rattle,
    perturbations.element_scaled_rattle,
    perturbations.stretch,
    perturbations.apply_perturbations,
    perturbations.Rattle(sigma=0.1),
    perturbations.ElementScaledRattle(sigma=0.1, reference={'H': 1.0}),
    perturbations.Stretch(hydro=0.1, shear=0.1),
    perturbations.Series((perturbations.Rattle(sigma=0.1), perturbations.Stretch(hydro=0.1, shear=0.1))),
    perturbations.RandomChoice(perturbations.Rattle(sigma=0.1), perturbations.Stretch(hydro=0.1, shear=0.1), chance=0.5),
    filters.AndFilter(filters.DistanceFilter({'H': 1.0}), filters.VolumeFilter(20.0)),
    filters.OrFilter(filters.DistanceFilter({'H': 1.0}), filters.VolumeFilter(20.0)),
    filters.DistanceFilter({'H': 1.0}),
    filters.AspectFilter(maximum_aspect_ratio=6),
    filters.VolumeFilter(maximum_volume_per_atom=20.0),
    filters.EnergyFilter(min_energy=-10.0, max_energy=0.0),
    filters.ForceFilter(max_force=1.0),
    calculators.Morse(epsilon=1.0, r0=1.0, rho0=1.0),
    pytest.param(calculators.Grace(model='GRACE-FS-OAM'), marks=pytest.mark.skipif(not HAS_TENSORPOTENTIAL, reason="tensorpotential not installed")),
    crystals.pyxtal,
    crystals.sample_space_groups,
    crystals.Formulas.range('H', 1, 3),
    neighbors.neighbor_list,
    plot.volume_histogram,
    plot.size_histogram,
    plot.concentration_histogram,
    plot.distance_histogram,
    plot.radial_distribution,
    plot.energy_histogram,
    plot.energy_volume,
    relax.Relax(max_steps=100, force_tolerance=1e-3),
    relax.CellRelax(max_steps=100, force_tolerance=1e-3),
    relax.VolumeRelax(max_steps=100, force_tolerance=1e-3, pressure=0.0),
    relax.SymmetryRelax(max_steps=100, force_tolerance=1e-3, pressure=0.0),
    relax.FullRelax(max_steps=100, force_tolerance=1e-3, pressure=0.0),
    relax.relax,
])
def test_pickle(obj):
    pickled = pickle.dumps(obj)
    unpickled = pickle.loads(pickled)

    # Check that they are of the same type
    assert type(unpickled) is type(obj)

    # For functions, we don't need further checks (identity might not hold if they are dynamically created, but these are module-level)
    if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
        assert unpickled.__name__ == obj.__name__
        return

    if isinstance(obj, type):
        assert unpickled == obj
        return

    # For instances
    assert assert_equal_recursive(unpickled, obj)
