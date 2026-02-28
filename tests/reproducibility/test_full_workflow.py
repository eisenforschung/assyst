import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from ase.calculators.morse import MorsePotential
from assyst.crystals import Formulas, sample_space_groups
from assyst.filters import DistanceFilter, AspectFilter, VolumeFilter
from assyst.relax import VolumeRelax, FullRelax, relax
from assyst.perturbations import RandomChoice, Rattle, Stretch, apply_perturbations

def run_workflow(rng):
    # 1. Sampling Random Structures
    max_num = 2
    fs = Formulas.range('Cu', max_num + 1)

    # Using a small subset of spacegroups to keep it fast but ensuring variety
    # group 225 is Fm-3m (cubic), group 194 is P6_3/mmc (hexagonal)
    spacegroups = [225, 194]

    spg = list(filter(
        AspectFilter(6),
        sample_space_groups(
            fs,
            spacegroups=spacegroups,
            max_atoms=max_num,
            rng=rng
        )
    ))

    # 2. Relaxing Configurations
    # Morse Potential for Cu (roughly)
    reference = MorsePotential(epsilon=.3, r0=2.55265548*1.10619396, rho0=4)

    volset = VolumeRelax(max_steps=5, force_tolerance=1e-2) # reduced steps for speed
    allset = FullRelax(max_steps=5, force_tolerance=1e-2)   # reduced steps for speed

    volmin = list(relax(volset, reference, spg))
    allmin = list(relax(allset, reference, volmin))

    # 3. Random Perturbations
    # Setup perturbations with the seed
    # Note: If we share the same 'rng' instance across multiple perturbation objects,
    # they will consume the same stream of random numbers.
    rattle_p = Rattle(.25, rng=rng) + Stretch(hydro=.05, shear=0.005, rng=rng)
    hydro = Stretch(hydro=.80, shear=.05, rng=rng)
    shear = Stretch(hydro=.05, shear=.20, rng=rng)
    stretch_p = RandomChoice(hydro, shear, .7, rng=rng)

    mods = [rattle_p, stretch_p] # apply each once

    random = list(apply_perturbations(allmin, mods, filters=[DistanceFilter({'Cu': 1})]))

    # 4. Final Combination and Filtering
    # Note: Calculating energies again would require attaching calculator to random structures
    # For this reproducibility test, we check geometry.

    everything = list(filter(VolumeFilter(300), filter(DistanceFilter({'Cu': 1}), spg + volmin + allmin + random)))

    return everything

@settings(deadline=None)
@given(st.integers(min_value=0, max_value=2**32-1))
def test_full_workflow_reproducibility(seed):
    # Run 1
    # We must explicitly construct the generator to pass it, ensuring we control the state
    rng1 = np.random.default_rng(seed)
    results1 = run_workflow(rng1)

    # Run 2
    # Resetting the generator with the same seed for the second run
    rng2 = np.random.default_rng(seed)
    results2 = run_workflow(rng2)

    # Comparison
    assert len(results1) == len(results2)
    # We can't guarantee > 0 for all random seeds without picking specific ones,
    # but with the chosen params it's highly likely.
    # If empty, they should both be empty, which satisfies reproducibility.

    for i, (s1, s2) in enumerate(zip(results1, results2)):
        assert len(s1) == len(s2), f"Structure {i} atom count mismatch"
        assert s1.get_chemical_formula() == s2.get_chemical_formula(), f"Structure {i} formula mismatch"

        # Check positions
        if not np.allclose(s1.positions, s2.positions, atol=1e-5):
            print(f"Structure {i} positions mismatch:")
            print("S1:", s1.positions)
            print("S2:", s2.positions)
            print("Diff:", s1.positions - s2.positions)
            assert False, f"Structure {i} positions mismatch"

        # Check cell
        if not np.allclose(s1.cell.array, s2.cell.array, atol=1e-5):
            print(f"Structure {i} cell mismatch:")
            print("S1:", s1.cell.array)
            print("S2:", s2.cell.array)
            assert False, f"Structure {i} cell mismatch"
