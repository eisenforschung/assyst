"""End-to-end test of leverage-score training set reduction on data derived from a universal GRACE model.

Runs the full ASSYST pipeline (symmetric structures -> relaxation -> perturbation), labels the resulting pool with
the universal GRACE foundation model, and reduces it with :mod:`assyst.leverage`.  The reference data is therefore
real MLIP output, not synthetic.

Requires `tensorpotential` and downloads the GRACE model on first run.
"""

import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from assyst.calculators import Grace
from assyst.crystals import Formulas, sample
from assyst.filters import DistanceFilter, VolumeFilter
from assyst.leverage import (
    RadialFeaturizer,
    configuration_leverage,
    design_matrix,
    select,
    stage_of,
    summarize,
    trace,
)
from assyst.perturbations import Rattle, Stretch, perturb
from assyst.relaxations import FullRelax, VolumeRelax, relax

try:
    import tensorpotential
except ImportError:
    tensorpotential = None

pytestmark = pytest.mark.skipif(tensorpotential is None, reason="tensorpotential not installed")

FEATURIZER = RadialFeaturizer(("Al",), n_radial=8, cutoff=6.0)
FRACTION = 0.3
"""Training fraction to reduce to; the paper reports plateau accuracy around 30-40%."""


def pipeline_step(structure):
    """Coarse ASSYST step of a structure, derived from the metadata ASSYST records on its own.

    Perturbations leave their parameters behind (``rattle(0.1)``, ``stretch(hydro=0.15, shear=0.05)``), which is
    more detail than we want here, and relaxed structures carry none at all.
    """
    perturbation = stage_of(structure)
    return "relaxed" if perturbation == "unperturbed" else perturbation.split("(")[0]


@pytest.fixture(scope="module")
def pool():
    """An ASSYST pool of Al structures labeled with the universal GRACE model."""
    calculator = Grace("GRACE-FS-OAM").get_calculator()

    seeds = list(
        sample(
            Formulas.range("Al", 1, 4),
            spacegroups=[1, 12, 62, 123, 139, 166, 194, 221, 225, 229],
            max_atoms=3,
            tolerance={"Al": 1.1},
            rng=42,
        )
    )
    settings = {"max_steps": 50, "force_tolerance": 1e-2}
    volume_minima = list(relax(seeds, VolumeRelax(**settings), calculator))
    minima = list(relax(volume_minima, FullRelax(**settings), calculator))

    filters = [DistanceFilter({"Al": 1.0}), VolumeFilter(6, 60)]
    rattled = list(
        perturb(
            minima,
            [Rattle(0.1, create_supercells=True, rng=1), Rattle(0.25, create_supercells=True, rng=2)],
            filters=filters,
        )
    )
    stretched = list(
        perturb(
            minima,
            [
                Stretch(hydro=0.15, shear=0.05, rng=3),
                Stretch(hydro=0.35, shear=0.08, rng=4),
                Stretch(hydro=0.55, shear=0.12, rng=5),
            ],
            filters=filters,
        )
    )

    labeled = []
    for structure in minima + rattled + stretched:
        structure = structure.copy()
        structure.calc = calculator
        energy, forces = structure.get_potential_energy(), structure.get_forces()
        structure.calc = SinglePointCalculator(structure, energy=energy, forces=forces)
        labeled.append(structure)
    return labeled


@pytest.fixture(scope="module")
def labels(pool):
    energies = np.array([s.get_potential_energy() for s in pool])
    return energies, np.array([len(s) for s in pool])


@pytest.fixture(scope="module")
def scores(pool):
    return configuration_leverage(pool, FEATURIZER)


def fit_errors(pool, labels, train):
    """Absolute per-atom energy errors over the whole pool of a ridge fit on the given subset."""
    energies, n_atoms = labels
    matrix, _ = design_matrix(pool, FEATURIZER)
    a, y = matrix[train], energies[train]
    coefficients = np.linalg.solve(a.T @ a + 1e-8 * np.eye(a.shape[1]), a.T @ y)
    return np.abs((matrix @ coefficients - energies) / n_atoms)


def test_pool_is_labeled_and_diverse(pool, labels):
    """The GRACE-labeled pool spans a wide range of structures, as ASSYST intends."""
    energies, n_atoms = labels
    assert len(pool) > 50, "Pipeline must produce a pool worth reducing!"
    assert {pipeline_step(s) for s in pool} == {"relaxed", "rattle", "stretch"}
    for structure in pool:
        assert isinstance(structure.calc, SinglePointCalculator)
        assert structure.get_forces().shape == (len(structure), 3)

    per_atom = energies / n_atoms
    assert per_atom.min() < -3.5, "Relaxed Al must reach roughly the GRACE cohesive energy!"
    assert per_atom.max() - per_atom.min() > 1.0, "Perturbations must span a wide energy range!"


def test_scores_are_label_free(pool, scores, labels):
    """Leverage scores use only feature geometry, so shuffling the reference energies cannot change them."""
    energies, _ = labels
    shuffled = []
    for structure, energy in zip(pool, energies[::-1]):
        structure = structure.copy()
        structure.calc = SinglePointCalculator(structure, energy=energy, forces=np.zeros((len(structure), 3)))
        shuffled.append(structure)
    np.testing.assert_allclose(
        configuration_leverage(shuffled, FEATURIZER),
        scores,
        atol=1e-12,
        err_msg="Scores must not depend on reference labels!",
    )


def test_relaxed_minima_are_redundant(pool, scores):
    """Relaxed minima sit in the densely sampled part of feature space, perturbed structures do not."""
    stages = np.array([pipeline_step(s) for s in pool])
    assert scores[stages == "relaxed"].mean() < scores[stages == "stretch"].mean(), \
        "Relaxed minima must carry less unique information than stretched structures!"
    ranked = np.argsort(scores)[::-1]
    assert stages[ranked[0]] != "relaxed", "The single most informative structure must not be a relaxed minimum!"


def test_selection_is_valid_and_reproducible(pool):
    """A reduction returns the requested number of distinct pool members, reproducibly."""
    selected = select(pool, fraction=FRACTION, featurizer=FEATURIZER, rng=0)
    assert len(selected) == int(np.ceil(FRACTION * len(pool)))
    assert len(set(selected.tolist())) == len(selected), "Selection must be without replacement!"
    assert set(selected.tolist()) <= set(range(len(pool)))
    np.testing.assert_array_equal(
        selected, select(pool, fraction=FRACTION, featurizer=FEATURIZER, rng=0),
        err_msg="Selection must be reproducible for a fixed seed!",
    )
    assert not np.array_equal(selected, select(pool, fraction=FRACTION, featurizer=FEATURIZER, rng=1)), \
        "Different seeds must give different draws!"


def test_block_mode_differs_from_energy_mode(pool):
    """Adding force rows (block-CUR) weights structures differently than energy-CUR."""
    energy_scores = configuration_leverage(pool, FEATURIZER)
    block_scores = configuration_leverage(pool, FEATURIZER, force_weight=0.1)
    assert not np.allclose(energy_scores / energy_scores.sum(), block_scores / block_scores.sum()), \
        "Force rows must change the relative weighting of structures!"
    selected = select(pool, fraction=FRACTION, featurizer=FEATURIZER, force_weight=0.1, rng=0)
    assert len(selected) == int(np.ceil(FRACTION * len(pool)))


def test_leverage_captures_distinctive_structures(pool, scores):
    """The point of the method: rare, informative structures are picked far more often than by chance.

    Random selection captures the most informative structures only at the sampling rate; leverage sampling must do
    substantially better, since that is what buys the reduction in labeling effort.
    """
    number = int(np.ceil(FRACTION * len(pool)))
    distinctive = set(np.argsort(scores)[::-1][:5].tolist())

    leverage = [select(pool, number=number, featurizer=FEATURIZER, rng=seed) for seed in range(20)]
    captured = np.mean([len(distinctive & set(s.tolist())) for s in leverage])
    uniform_expectation = 5 * number / len(pool)
    assert captured > 2.0 * uniform_expectation, (
        f"Leverage sampling must strongly enrich the most informative structures "
        f"({captured:.2f} of 5 captured vs {uniform_expectation:.2f} expected by chance)!"
    )


def test_leverage_covers_pool_better_than_random(pool, labels):
    """A leverage-reduced training set reproduces the whole pool better than a random one of the same size.

    Compares the typical and the tail behaviour: random draws occasionally miss the extreme structures entirely and
    then extrapolate badly, which is exactly what leverage sampling protects against.
    """
    number = int(np.ceil(FRACTION * len(pool)))
    leverage = np.array([
        fit_errors(pool, labels, select(pool, number=number, featurizer=FEATURIZER, rng=seed)).max()
        for seed in range(20)
    ])
    rng = np.random.default_rng(0)
    random = np.array([
        fit_errors(pool, labels, rng.choice(len(pool), size=number, replace=False)).max()
        for _ in range(100)
    ])
    assert np.median(leverage) < np.median(random), (
        f"Leverage selection must typically cover the pool better than random "
        f"({np.median(leverage):.3f} vs {np.median(random):.3f} eV/atom worst-case error)!"
    )
    assert np.quantile(leverage, 0.9) < np.quantile(random, 0.9), (
        f"Leverage selection must avoid the bad draws random selection makes "
        f"({np.quantile(leverage, 0.9):.3f} vs {np.quantile(random, 0.9):.3f} eV/atom)!"
    )


def test_reduced_set_retains_accuracy(pool, labels):
    """Dropping most of the pool must cost little: the reduced fit stays close to the full-pool fit."""
    full = fit_errors(pool, labels, np.arange(len(pool)))
    number = int(np.ceil(FRACTION * len(pool)))
    reduced = np.array([
        np.sqrt((fit_errors(pool, labels, select(pool, number=number, featurizer=FEATURIZER, rng=seed)) ** 2).mean())
        for seed in range(20)
    ])
    full_rmse = np.sqrt((full**2).mean())
    assert np.median(reduced) < 2.0 * full_rmse, (
        f"A {FRACTION:.0%} leverage subset must nearly recover the full-pool fit "
        f"({np.median(reduced):.3f} vs {full_rmse:.3f} eV/atom RMSE)!"
    )


# --- provenance tracing ---

def test_trace_accounts_for_every_structure(pool, scores):
    """Every pool member is traced to its ASSYST step and marked either selected or discarded."""
    selected = select(pool, fraction=FRACTION, featurizer=FEATURIZER, rng=0)
    traced = trace(pool, selected, scores=scores, stage=pipeline_step)

    assert len(traced) == len(pool)
    assert traced["selected"].sum() == len(selected)
    np.testing.assert_array_equal(
        np.flatnonzero(traced["selected"].to_numpy()), np.sort(selected),
        err_msg="Traced selection must be exactly the returned indices!",
    )
    ranks = traced["rank"].to_numpy()
    np.testing.assert_array_equal(
        ranks[selected], np.arange(len(selected)),
        err_msg="The structure drawn at position p must carry rank p!",
    )
    assert (traced.loc[~traced["selected"], "rank"] == -1).all(), "Discarded structures must have rank -1!"
    assert set(traced["stage"]) == {"relaxed", "rattle", "stretch"}
    np.testing.assert_allclose(traced["score"], scores)
    np.testing.assert_array_equal(traced["number_of_atoms"], [len(s) for s in pool])


def test_summary_balances(pool, scores):
    """The per-step summary adds up and reports where the reduction takes its structures from."""
    selected = select(pool, fraction=FRACTION, featurizer=FEATURIZER, rng=0)
    summary = summarize(trace(pool, selected, scores=scores, stage=pipeline_step))

    assert set(summary.index) == {"relaxed", "rattle", "stretch"}
    assert summary["pool"].sum() == len(pool)
    assert summary["selected"].sum() == len(selected)
    assert (summary["selected"] + summary["discarded"] == summary["pool"]).all()
    np.testing.assert_allclose(summary["selected_fraction"], summary["selected"] / summary["pool"])
    assert abs(summary["score_share"].sum() - 1.0) < 1e-12, "Score shares must partition the pool's leverage!"

    assert summary.loc["relaxed", "mean_score"] < summary.loc["stretch", "mean_score"], \
        "Relaxed minima must score below stretched structures!"
    assert (summary["discarded"] > 0).all(), \
        "A reduction to a third of the pool must discard structures from every step!"


def test_selection_spreads_over_all_steps(pool, scores):
    """The reduction keeps structures from every ASSYST step, rather than collapsing onto one."""
    selected = select(pool, fraction=FRACTION, featurizer=FEATURIZER, rng=0)
    summary = summarize(trace(pool, selected, scores=scores, stage=pipeline_step))
    assert (summary["selected"] > 0).all(), "Every ASSYST step must contribute to the reduced set!"
