import numpy as np
import pytest
from ase.calculators.morse import MorsePotential
from assyst.crystals import Formulas, sample_space_groups
from assyst.filters import DistanceFilter, AspectFilter, VolumeFilter
from assyst.relax import VolumeRelax, FullRelax, relax
from assyst.perturbations import RandomChoice, Rattle, Stretch, apply_perturbations

def run_workflow(seed):
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
            rng=seed
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
    # Important: We must create a new RNG for the perturbations to ensure they start
    # from the same state in each run_workflow call.
    rng = np.random.default_rng(seed)

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

def test_full_workflow_reproducibility():
    seed = 42

    # Run 1
    results1 = run_workflow(seed)

    # Run 2
    results2 = run_workflow(seed)

    # Comparison
    assert len(results1) == len(results2)
    assert len(results1) > 0 # Ensure we actually produced something

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
