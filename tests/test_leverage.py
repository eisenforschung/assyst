import numpy as np
import pytest
from ase.build import bulk

try:
    import pyace
except ImportError:
    pyace = None

from assyst.leverage import (
    AceFeaturizer,
    Featurizer,
    RadialFeaturizer,
    configuration_leverage,
    design_matrix,
    leverage_scores,
    sample_without_replacement,
    select,
    stage_of,
    summarize,
    trace,
)
from assyst.perturbations import Rattle, Stretch


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def copper(rng):
    structure = bulk("Cu", cubic=True).repeat((2, 1, 1))
    structure.rattle(stdev=0.1, rng=rng)
    return structure


@pytest.fixture
def salt(rng):
    structure = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    structure.rattle(stdev=0.1, rng=rng)
    return structure


# --- leverage_scores ---

def brute_force_scores(matrix, regularization, tikhonov=None):
    if tikhonov is None:
        tikhonov = np.ones(matrix.shape[1])
    hat = matrix @ np.linalg.solve(
        matrix.T @ matrix + regularization * np.diag(tikhonov**2), matrix.T
    )
    return np.diag(hat)


def test_leverage_scores_match_hat_matrix(rng):
    matrix = rng.normal(size=(20, 5))
    for regularization in (1e-6, 1e-2, 1.0):
        np.testing.assert_allclose(
            leverage_scores(matrix, regularization=regularization),
            brute_force_scores(matrix, regularization),
            atol=1e-12,
            err_msg="Leverage scores must equal the diagonal of the regularized hat matrix!",
        )


def test_leverage_scores_tikhonov(rng):
    matrix = rng.normal(size=(20, 5))
    tikhonov = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        leverage_scores(matrix, regularization=0.1, tikhonov=tikhonov),
        brute_force_scores(matrix, 0.1, tikhonov),
        atol=1e-12,
        err_msg="Tikhonov vector must act as a diagonal regularization operator!",
    )


def test_leverage_scores_classical(rng):
    matrix = rng.normal(size=(20, 5))
    scores = leverage_scores(matrix, regularization=0)
    np.testing.assert_allclose(
        scores,
        np.diag(matrix @ np.linalg.pinv(matrix)),
        atol=1e-12,
        err_msg="Unregularized scores must be the diagonal of the classical hat matrix!",
    )
    assert abs(scores.sum() - 5) < 1e-12, "Classical scores must sum to the rank of the matrix!"


def test_leverage_scores_rank_deficient(rng):
    matrix = rng.normal(size=(20, 5))
    matrix[:, 4] = matrix[:, 0]
    scores = leverage_scores(matrix, regularization=0)
    assert abs(scores.sum() - 4) < 1e-9, "Scores of a rank-deficient matrix must sum to its rank!"


def test_leverage_scores_bounds(rng):
    matrix = rng.normal(size=(30, 8))
    for regularization in (0, 1e-8, 1e-2):
        scores = leverage_scores(matrix, regularization=regularization)
        assert (scores >= 0).all() and (scores <= 1 + 1e-12).all(), "Scores must lie in [0, 1]!"


def test_leverage_scores_trace(rng):
    matrix = rng.normal(size=(30, 8))
    regularization = 0.1
    s = np.linalg.svd(matrix, compute_uv=False)
    assert abs(
        leverage_scores(matrix, regularization=regularization).sum()
        - (s**2 / (s**2 + regularization)).sum()
    ) < 1e-12, "Scores must sum to the effective degrees of freedom!"


def test_leverage_scores_duplicate_rows(rng):
    matrix = rng.normal(size=(10, 4))
    matrix[5] = matrix[3]
    scores = leverage_scores(matrix, regularization=1e-3)
    assert abs(scores[3] - scores[5]) < 1e-12, "Identical rows must have identical scores!"


def test_leverage_scores_zero_matrix():
    for regularization in (0, 1e-3):
        np.testing.assert_array_equal(
            leverage_scores(np.zeros((5, 3)), regularization=regularization),
            np.zeros(5),
            err_msg="Zero rows must have zero score!",
        )


def test_leverage_scores_input_validation(rng):
    matrix = rng.normal(size=(10, 4))
    with pytest.raises(ValueError):
        leverage_scores(np.zeros(5))
    with pytest.raises(ValueError):
        leverage_scores(matrix, regularization=-1)
    with pytest.raises(ValueError):
        leverage_scores(matrix, tikhonov=np.ones(3))
    with pytest.raises(ValueError):
        leverage_scores(matrix, tikhonov=-np.ones(4))


# --- RadialFeaturizer ---

def test_featurizer_validation():
    with pytest.raises(ValueError):
        RadialFeaturizer(())
    with pytest.raises(ValueError):
        RadialFeaturizer(("Cu", "Cu"))
    with pytest.raises(ValueError):
        RadialFeaturizer(("Cu",), n_radial=0)
    with pytest.raises(ValueError):
        RadialFeaturizer(("Cu",), cutoff=0.0)


def test_featurizer_unknown_element(copper):
    with pytest.raises(ValueError):
        RadialFeaturizer(("Al",))(copper)


def test_featurizer_shape(copper, salt):
    featurizer = RadialFeaturizer(("Cu",), n_radial=4)
    assert featurizer.n_features == 1 + 4
    assert featurizer(copper).shape == (len(copper), featurizer.n_features)
    featurizer = RadialFeaturizer(("Cl", "Na"), n_radial=3)
    assert featurizer.n_features == 2 * (1 + 2 * 3)
    assert featurizer(salt).shape == (len(salt), featurizer.n_features)
    assert featurizer.gradient(salt).shape == (len(salt), 3, featurizer.n_features)


def test_featurizer_constant_channel(salt):
    featurizer = RadialFeaturizer(("Cl", "Na"), n_radial=2)
    block = 1 + 2 * 2
    totals = featurizer(salt).sum(axis=0)
    assert totals[0 * block] == 4.0, "Constant channel must count Cl atoms!"
    assert totals[1 * block] == 4.0, "Constant channel must count Na atoms!"


def test_featurizer_species_blocks(salt):
    featurizer = RadialFeaturizer(("Cl", "Na"), n_radial=2)
    features = featurizer(salt)
    block = 1 + 2 * 2
    chlorine = np.array([s == "Cl" for s in salt.symbols])
    assert (features[chlorine, block:] == 0).all(), "Cl atoms must only fill the Cl feature block!"
    assert (features[~chlorine, :block] == 0).all(), "Na atoms must only fill the Na feature block!"


def test_featurizer_isolated_atom():
    from ase import Atoms

    atom = Atoms("Cu", cell=[20, 20, 20], pbc=True)
    featurizer = RadialFeaturizer(("Cu",), n_radial=3, cutoff=4.0)
    np.testing.assert_array_equal(featurizer(atom), [[1.0, 0, 0, 0]])
    np.testing.assert_array_equal(featurizer.gradient(atom), np.zeros((1, 3, 4)))


def test_featurizer_permutation_invariance(salt, rng):
    featurizer = RadialFeaturizer(("Cl", "Na"), n_radial=3)
    permutation = rng.permutation(len(salt))
    np.testing.assert_allclose(
        featurizer(salt[permutation]),
        featurizer(salt)[permutation],
        atol=1e-12,
        err_msg="Per-atom features must permute with the atoms!",
    )


def test_featurizer_translation_invariance(copper):
    featurizer = RadialFeaturizer(("Cu",), n_radial=3)
    translated = copper.copy()
    translated.positions += [0.31, -1.7, 0.05]
    translated.wrap()
    np.testing.assert_allclose(
        featurizer(translated).sum(axis=0),
        featurizer(copper).sum(axis=0),
        atol=1e-10,
        err_msg="Summed features must be invariant under rigid translations!",
    )


@pytest.mark.parametrize("fixture", ["copper", "salt"])
def test_featurizer_gradient_finite_differences(fixture, request):
    structure = request.getfixturevalue(fixture)
    featurizer = RadialFeaturizer(tuple(sorted(set(structure.symbols))), n_radial=3, cutoff=4.0)
    gradient = featurizer.gradient(structure)
    step = 1e-5
    for atom in range(len(structure)):
        for direction in range(3):
            plus, minus = structure.copy(), structure.copy()
            plus.positions[atom, direction] += step
            minus.positions[atom, direction] -= step
            finite_difference = (featurizer(plus).sum(axis=0) - featurizer(minus).sum(axis=0)) / (2 * step)
            np.testing.assert_allclose(
                gradient[atom, direction],
                finite_difference,
                atol=1e-6,
                err_msg="Analytic gradient must match finite differences!",
            )


# --- design_matrix ---

def test_design_matrix_energy_rows(copper):
    featurizer = RadialFeaturizer(("Cu",), n_radial=3)
    structures = [copper, copper.repeat((1, 2, 1))]
    matrix, owners = design_matrix(structures, featurizer, energy_weight=2.0)
    assert matrix.shape == (2, featurizer.n_features)
    np.testing.assert_array_equal(owners, [0, 1])
    np.testing.assert_allclose(
        matrix[0],
        2.0 * featurizer(copper).sum(axis=0),
        err_msg="Energy row must be the weighted sum of per-atom features!",
    )


def test_design_matrix_force_rows(copper):
    featurizer = RadialFeaturizer(("Cu",), n_radial=3)
    matrix, owners = design_matrix([copper], featurizer, force_weight=0.1)
    assert matrix.shape == (1 + 3 * len(copper), featurizer.n_features)
    assert (owners == 0).all()
    np.testing.assert_allclose(
        matrix[1:],
        -0.1 * featurizer.gradient(copper).reshape(3 * len(copper), -1),
        err_msg="Force rows must be the weighted negative feature gradients!",
    )


def test_design_matrix_empty():
    with pytest.raises(ValueError):
        design_matrix([], RadialFeaturizer(("Cu",)))


class CountFeaturizer(Featurizer):
    """Minimal custom featurizer: one constant channel per atom, no gradients."""

    def __call__(self, structure):
        return np.ones((len(structure), 1))


def test_custom_featurizer(copper):
    """Any Featurizer subclass plugs in; only force rows need gradients."""
    structures = [copper, copper.repeat((1, 2, 1))]
    matrix, owners = design_matrix(structures, CountFeaturizer())
    np.testing.assert_array_equal(matrix, [[len(copper)], [2 * len(copper)]])
    np.testing.assert_array_equal(owners, [0, 1])
    assert len(select(structures, number=1, featurizer=CountFeaturizer(), rng=0)) == 1


def test_featurizer_without_gradient_rejects_force_rows(copper):
    with pytest.raises(NotImplementedError, match="CountFeaturizer"):
        design_matrix([copper], CountFeaturizer(), force_weight=0.1)


# --- AceFeaturizer ---

requires_pyace = pytest.mark.skipif(pyace is None, reason="pyace not installed")


@requires_pyace
def test_ace_validation():
    with pytest.raises(ValueError):
        AceFeaturizer(())
    with pytest.raises(ValueError):
        AceFeaturizer(("Al", "Al"))
    with pytest.raises(ValueError):
        AceFeaturizer(("Al",), n_radial=(4, 2), l_max=(0, 2, 2))
    with pytest.raises(ValueError):
        AceFeaturizer(("Al",), n_radial=(), l_max=())
    with pytest.raises(ValueError):
        AceFeaturizer(("Al",), n_radial=(0,), l_max=(0,))
    with pytest.raises(ValueError):
        AceFeaturizer(("Al",), n_radial=(4, 2), l_max=(0, -1))
    with pytest.raises(ValueError):
        AceFeaturizer(("Al",), n_radial=(4, 2), l_max=(1, 2))
    with pytest.raises(ValueError):
        AceFeaturizer(("Al",), cutoff=0.0)
    with pytest.raises(ValueError):
        AceFeaturizer(("Al",), n_functions_per_element=0)
    with pytest.raises(ValueError, match="together"):
        AceFeaturizer(("Al",), n_radial=(4, 2))
    with pytest.raises(ValueError, match="together"):
        AceFeaturizer(("Al",), l_max=(0, 2))
    with pytest.raises(FileNotFoundError):
        AceFeaturizer(("Al",), potential="does-not-exist.yaml")
    with pytest.raises(FileNotFoundError):
        AceFeaturizer.from_potential("does-not-exist.yaml")


@requires_pyace
def test_ace_defaults_match_pacemaker_template():
    """The default basis must be the one ``pacemaker -t`` writes, with no functions_per_element filter."""
    featurizer = AceFeaturizer(("Al",))
    assert featurizer.n_radial is None and featurizer.l_max is None, "Default must defer to the template blocks!"
    assert featurizer.cutoff == 7.0
    assert featurizer.radial_base == "SBessel"
    assert featurizer.radial_parameters == (5.25,)
    assert featurizer.n_functions_per_element is None, "The template's filter must be off by default!"
    # the untruncated template basis; a change here means the pyace defaults moved
    assert featurizer.n_features == 945
    assert AceFeaturizer(("Al", "Cu")).n_features == 4452


@requires_pyace
def test_ace_functions_per_element_truncates():
    assert AceFeaturizer(("Al",), n_functions_per_element=200).n_features == 200
    assert AceFeaturizer(("Al",)).n_features > 200, "Truncation must actually drop functions!"


@requires_pyace
def test_ace_from_potential(tmp_path, aluminium):
    """A featurizer built from a saved basis must reproduce the basis it was saved from."""
    from assyst.leverage import _ace_engine

    source = AceFeaturizer(("Al",), n_radial=(6, 3, 2), l_max=(0, 2, 2), cutoff=6.0)
    path = tmp_path / "potential.yaml"
    _ace_engine(source)[0].save(str(path))

    loaded = AceFeaturizer.from_potential(path)
    assert loaded.elements == ("Al",), "Elements must come from the potential!"
    assert loaded.potential == str(path)
    assert loaded.n_features == source.n_features
    assert np.allclose(loaded(aluminium), source(aluminium))
    assert np.allclose(loaded.gradient(aluminium), source.gradient(aluminium))


@requires_pyace
def test_ace_from_potential_rejects_element_mismatch(tmp_path):
    """Declaring elements the potential was not fitted for must not silently mislabel the feature blocks."""
    from assyst.leverage import _ace_engine

    source = AceFeaturizer(("Al",), n_radial=(4, 2), l_max=(0, 2), cutoff=6.0)
    path = tmp_path / "potential.yaml"
    _ace_engine(source)[0].save(str(path))

    mismatched = AceFeaturizer(("Al", "Cu"), potential=str(path))
    with pytest.raises(ValueError, match="fitted for elements"):
        _ = mismatched.n_features


@requires_pyace
def test_ace_from_potential_rejects_non_basis(tmp_path):
    path = tmp_path / "not-a-potential.yaml"
    path.write_text("hello: world\n")
    with pytest.raises(ValueError, match="C-tilde"):
        AceFeaturizer.from_potential(path)


@pytest.fixture
def ace():
    return AceFeaturizer(("Al",), n_radial=(6, 3, 2), l_max=(0, 2, 2), cutoff=6.0)


@pytest.fixture
def aluminium(rng):
    structure = bulk("Al", cubic=True).repeat((2, 1, 1))
    structure.rattle(stdev=0.1, rng=rng)
    return structure


@requires_pyace
def test_ace_shapes(ace, aluminium):
    assert ace.n_features > len(ace.n_radial), "The basis must hold more than one function per body order!"
    assert ace(aluminium).shape == (len(aluminium), ace.n_features)
    assert ace.gradient(aluminium).shape == (len(aluminium), 3, ace.n_features)


@requires_pyace
def test_ace_matches_pyace_design_row(ace, aluminium):
    """The summed per-atom features must be exactly pyace's own energy design row."""
    import pyace.linearacefit as plf
    from pyace.atomicenvironment import aseatoms_to_atomicenvironment

    from assyst.leverage import _ace_engine

    basis, _, calculator, n_features, shifts = _ace_engine(ace)
    plf.g_calc, plf.g_func_ind_shift, plf.g_nfunc = calculator, shifts, n_features
    environment = aseatoms_to_atomicenvironment(
        aluminium, cutoff=basis.cutoffmax, elements_mapper_dict=basis.elements_to_index_map
    )
    energy_row, _ = plf.compute_b_grad_ae(environment)
    np.testing.assert_allclose(
        ace(aluminium).sum(axis=0), energy_row, atol=1e-12,
        err_msg="Summed ACE features must reproduce pyace's energy design row!",
    )


@requires_pyace
def test_ace_gradient_finite_differences(ace, aluminium):
    gradient = ace.gradient(aluminium)
    step = 1e-5
    for atom in range(len(aluminium)):
        for direction in range(3):
            plus, minus = aluminium.copy(), aluminium.copy()
            plus.positions[atom, direction] += step
            minus.positions[atom, direction] -= step
            finite_difference = (ace(plus).sum(axis=0) - ace(minus).sum(axis=0)) / (2 * step)
            np.testing.assert_allclose(
                gradient[atom, direction], finite_difference, atol=1e-5,
                err_msg="Analytic ACE gradient must match finite differences!",
            )


@requires_pyace
def test_ace_species_blocks():
    """Each element gets its own block of basis functions."""
    featurizer = AceFeaturizer(("Al", "Cu"), n_radial=(4, 2), l_max=(0, 2))
    structure = bulk("AlCu", "rocksalt", a=5.0, cubic=True)
    features = featurizer(structure)
    aluminium = np.array([s == "Al" for s in structure.symbols])
    for mask in (aluminium, ~aluminium):
        used = np.flatnonzero((features[mask] != 0).any(axis=0))
        other = np.flatnonzero((features[~mask] != 0).any(axis=0))
        assert not set(used) & set(other), "Species blocks must not overlap!"


@requires_pyace
def test_ace_unknown_element(ace):
    with pytest.raises(ValueError, match="Cu"):
        ace(bulk("Cu"))


@requires_pyace
def test_ace_picklable(ace, aluminium):
    import pickle

    restored = pickle.loads(pickle.dumps(ace))
    assert restored == ace
    np.testing.assert_allclose(restored(aluminium), ace(aluminium))


@requires_pyace
def test_ace_drives_selection(ace, aluminium):
    pool = [aluminium, aluminium.repeat((1, 2, 1)), bulk("Al", cubic=True)]
    matrix, owners = design_matrix(pool, ace, force_weight=0.1)
    assert matrix.shape[1] == ace.n_features
    assert len(owners) == len(pool) + 3 * sum(len(s) for s in pool)
    selected = select(pool, number=2, featurizer=ace, rng=0)
    assert len(set(selected.tolist())) == 2


def test_design_matrix_fits_pair_potential(rng):
    """The featurizer spans pair potentials, so a linear fit on the design matrix must reproduce one."""
    from ase.calculators.morse import MorsePotential

    structures = []
    for _ in range(30):
        s = bulk("Cu", cubic=True, a=3.6 * rng.uniform(0.97, 1.15))
        s.rattle(stdev=0.05, rng=rng)
        structures.append(s)
    # ASE's Morse potential cuts off at rcut2 * r0 = 2.7 * 2.5
    featurizer = RadialFeaturizer(("Cu",), n_radial=12, cutoff=6.75)
    energies = []
    forces = []
    for s in structures:
        s.calc = MorsePotential(epsilon=0.3, r0=2.5, rho0=4)
        energies.append(s.get_potential_energy())
        forces.append(s.get_forces())

    matrix, _ = design_matrix(structures, featurizer, force_weight=0.1)
    targets = np.concatenate(
        [np.concatenate(([e], 0.1 * f.ravel())) for e, f in zip(energies, forces)]
    )
    coefficients = np.linalg.lstsq(matrix, targets, rcond=None)[0]
    residuals = matrix @ coefficients - targets
    energy_rows = np.concatenate([[True] + [False] * (3 * len(s)) for s in structures])
    assert np.sqrt((residuals[energy_rows] ** 2).mean()) < 1e-3, \
        "Linear fit on the design matrix must reproduce a pair potential's energies!"
    assert np.sqrt((residuals[~energy_rows] ** 2).mean()) / 0.1 < 5e-3, \
        "Linear fit on the design matrix must reproduce a pair potential's forces!"


# --- configuration_leverage ---

def test_configuration_leverage_shape_and_sign(copper):
    featurizer = RadialFeaturizer(("Cu",), n_radial=3)
    structures = [copper, copper.repeat((1, 2, 1)), copper.repeat((1, 1, 2))]
    scores = configuration_leverage(structures, featurizer)
    assert scores.shape == (3,)
    assert (scores > 0).all(), "Energy rows always carry the constant channel, so scores must be positive!"


def test_configuration_leverage_outlier(rng):
    """A structure far from an otherwise homogeneous pool must get the highest score."""
    pool = []
    for _ in range(10):
        s = bulk("Cu", cubic=True)
        s.rattle(stdev=0.01, rng=rng)
        pool.append(s)
    outlier = bulk("Cu", cubic=True, a=4.7)
    outlier.rattle(stdev=0.3, rng=rng)
    pool.append(outlier)
    featurizer = RadialFeaturizer(("Cu",), n_radial=4)
    for force_weight in (None, 0.1):
        scores = configuration_leverage(pool, featurizer, force_weight=force_weight)
        assert scores.argmax() == len(pool) - 1, "The outlier structure must have the highest leverage!"


# --- sample_without_replacement ---

def test_sample_reproducible():
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    first = sample_without_replacement(weights, 3, rng=123)
    second = sample_without_replacement(weights, 3, rng=123)
    np.testing.assert_array_equal(first, second, err_msg="Equal seeds must give equal draws!")
    third = sample_without_replacement(weights, 3, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(first, third, err_msg="Seed and equally seeded generator must agree!")


def test_sample_unique_and_in_range():
    weights = np.arange(1.0, 21.0)
    drawn = sample_without_replacement(weights, 20, rng=0)
    assert sorted(drawn) == list(range(20)), "Drawing everything must return every index exactly once!"


def test_sample_skips_zero_weights():
    weights = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    for seed in range(20):
        drawn = sample_without_replacement(weights, 2, rng=seed)
        assert set(drawn) == {1, 3}, "Zero weight structures must never be drawn!"
    with pytest.raises(ValueError):
        sample_without_replacement(weights, 3)


def test_sample_prefers_heavy_weights():
    weights = np.array([1e12, 1.0, 1.0, 1.0])
    for seed in range(20):
        assert sample_without_replacement(weights, 1, rng=seed)[0] == 0, \
            "An overwhelming weight must dominate the first draw!"


def test_sample_input_validation():
    with pytest.raises(ValueError):
        sample_without_replacement(np.array([1.0, -1.0]), 1)
    with pytest.raises(ValueError):
        sample_without_replacement(np.array([[1.0]]), 1)
    with pytest.raises(ValueError):
        sample_without_replacement(np.ones(3), 4)
    with pytest.raises(ValueError):
        sample_without_replacement(np.ones(3), -1)


# --- select ---

@pytest.fixture
def pool(rng):
    structures = []
    for element, lattice_constant in (("Cu", 3.6), ("Al", 4.05)):
        for _ in range(10):
            s = bulk(element, "fcc", a=lattice_constant * rng.uniform(0.95, 1.1), cubic=True)
            s.rattle(stdev=0.05, rng=rng)
            structures.append(s)
    return structures


def test_select_fraction(pool):
    indices = select(pool, fraction=0.3, rng=0)
    assert len(indices) == 6, "fraction must be rounded up to full structures!"
    assert len(set(indices)) == len(indices), "Selection must be without replacement!"
    assert all(0 <= i < len(pool) for i in indices)
    np.testing.assert_array_equal(
        indices, select(pool, fraction=0.3, rng=0), err_msg="Selection must be reproducible!"
    )


def test_select_number(pool):
    assert len(select(pool, number=5, rng=0)) == 5
    assert sorted(select(pool, number=len(pool), rng=0)) == list(range(len(pool)))


def test_select_block_mode(pool):
    featurizer = RadialFeaturizer(("Al", "Cu"), n_radial=4)
    energy_scores = configuration_leverage(pool, featurizer)
    block_scores = configuration_leverage(pool, featurizer, force_weight=0.1)
    assert not np.allclose(energy_scores, block_scores), "Force rows must change the scores!"
    indices = select(pool, fraction=0.25, force_weight=0.1, rng=1)
    assert len(indices) == 5


def test_select_input_validation(pool):
    with pytest.raises(ValueError):
        select(pool)
    with pytest.raises(ValueError):
        select(pool, fraction=0.5, number=3)
    with pytest.raises(ValueError):
        select(pool, fraction=0.0)
    with pytest.raises(ValueError):
        select(pool, fraction=1.5)


# --- provenance tracing ---

def test_stage_of_reads_assyst_metadata(copper):
    """Stages come from the metadata ASSYST already records, not from anything the reduction adds."""
    assert stage_of(copper) == "unperturbed", "Structures without a perturbation are generated or relaxed ones!"
    assert stage_of(Rattle(0.05)(copper.copy())) == "rattle(0.05)"
    assert stage_of(Stretch(hydro=0.1, shear=0.02)(copper.copy())) == "stretch(hydro=0.1, shear=0.02)"
    chained = Stretch(hydro=0.1, shear=0.02)(Rattle(0.05)(copper.copy()))
    assert stage_of(chained) == "rattle(0.05)+stretch(hydro=0.1, shear=0.02)", \
        "Chained perturbations must be reported in full!"


@pytest.fixture
def tagged_pool(pool):
    """Half relaxed structures, half rattled ones, distinguished only by ASSYST's own metadata."""
    return pool[:10] + [Rattle(0.05, rng=i)(s.copy()) for i, s in enumerate(pool[10:])]


def test_trace_columns_and_accounting(tagged_pool):
    featurizer = RadialFeaturizer(("Al", "Cu"), n_radial=4)
    selected = select(tagged_pool, number=6, featurizer=featurizer, rng=0)
    traced = trace(tagged_pool, selected, featurizer=featurizer)

    assert list(traced.columns) == [
        "stage", "selected", "rank", "score", "number_of_atoms", "volume_per_atom", "formula"
    ]
    assert len(traced) == len(tagged_pool)
    assert traced["selected"].sum() == 6
    np.testing.assert_array_equal(traced["rank"].to_numpy()[selected], np.arange(6))
    assert (traced.loc[~traced["selected"], "rank"] == -1).all()
    np.testing.assert_array_equal(np.flatnonzero(traced["selected"].to_numpy()), np.sort(selected))
    assert set(traced["stage"]) == {"unperturbed", "rattle(0.05)"}


def test_trace_accepts_explicit_scores(tagged_pool):
    scores = np.arange(len(tagged_pool), dtype=float) + 1
    traced = trace(tagged_pool, [0, 1], scores=scores)
    np.testing.assert_array_equal(traced["score"], scores)


def test_trace_custom_stage_function(tagged_pool):
    traced = trace(tagged_pool, [0], scores=np.ones(len(tagged_pool)),
                   stage=lambda s: s.get_chemical_formula())
    assert set(traced["stage"]) == {s.get_chemical_formula() for s in tagged_pool}


def test_trace_score_validation(tagged_pool):
    with pytest.raises(ValueError):
        trace(tagged_pool, [0], scores=np.ones(3))


def test_summarize_balances(tagged_pool):
    featurizer = RadialFeaturizer(("Al", "Cu"), n_radial=4)
    selected = select(tagged_pool, number=8, featurizer=featurizer, rng=0)
    summary = summarize(trace(tagged_pool, selected, featurizer=featurizer))

    assert list(summary.columns) == [
        "pool", "selected", "discarded", "selected_fraction", "mean_score", "score_share"
    ]
    assert set(summary.index) == {"unperturbed", "rattle(0.05)"}
    assert summary["pool"].sum() == len(tagged_pool)
    assert summary["selected"].sum() == len(selected)
    assert (summary["selected"] + summary["discarded"] == summary["pool"]).all()
    np.testing.assert_allclose(summary["selected_fraction"], summary["selected"] / summary["pool"])
    assert abs(summary["score_share"].sum() - 1.0) < 1e-12, "Score shares must sum to one!"


def test_summarize_counts_match_stages(tagged_pool):
    summary = summarize(trace(tagged_pool, [0, 1, 2], scores=np.ones(len(tagged_pool))))
    assert summary.loc["unperturbed", "pool"] == 10
    assert summary.loc["rattle(0.05)", "pool"] == len(tagged_pool) - 10
    assert summary.loc["unperturbed", "selected"] == 3
    assert summary.loc["rattle(0.05)", "selected"] == 0
    assert summary.loc["rattle(0.05)", "discarded"] == len(tagged_pool) - 10
