"""The ``step`` key records which workflow step produced a structure."""

import pytest
from ase import Atoms
from ase.build import bulk

from assyst.calculators import Morse
from assyst.crystals import Formulas, pyxtal, sample
from assyst.perturbations import Rattle, Series, Stretch, perturb, rattle
from assyst.relaxations import CellRelax, FullRelax, Relax, SymmetryRelax, VolumeRelax, relax
from assyst.utils import STEP_KEY, step_of, update_uuid


@pytest.fixture
def cu():
    s = bulk("Cu", cubic=True)
    s.calc = Morse().get_calculator()
    return s


@pytest.fixture
def cu2():
    return Atoms("Cu2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)


# --- update_uuid / step_of ---


def test_update_uuid_sets_key():
    s = Atoms("H")
    update_uuid(s, step="pyxtal")
    assert s.info[STEP_KEY] == "pyxtal"


def test_update_uuid_replaces_the_previous_step():
    """Only the most recent step is kept, like ``uuid`` itself."""
    s = Atoms("H")
    update_uuid(s, step="volume_relax")
    update_uuid(s, step="full_relax")
    assert s.info[STEP_KEY] == "full_relax"


def test_step_of_reads_the_key():
    s = Atoms("H")
    s.info[STEP_KEY] = "volume_relax"
    assert step_of(s) == "volume_relax"


def test_step_of_defaults_for_foreign_structures():
    assert step_of(Atoms("H")) == "unknown"
    assert step_of(Atoms("H"), default="external") == "external"


# --- relaxations ---


@pytest.mark.parametrize(
    "settings, name",
    [
        (Relax, "relax"),
        (CellRelax, "cell_relax"),
        (VolumeRelax, "volume_relax"),
        (SymmetryRelax, "symmetry_relax"),
        (FullRelax, "full_relax"),
    ],
)
def test_relaxation_records_its_own_name(settings, name, cu):
    assert str(settings()) == name
    assert step_of(settings(max_steps=1).relax(cu)) == name


def test_relaxation_name_ignores_optimizer_settings():
    """How hard the optimizer works does not change what the structure is."""
    assert str(FullRelax(max_steps=5, force_tolerance=1e-1, algorithm="FIRE")) == str(FullRelax())


@pytest.mark.parametrize(
    "settings, name",
    [
        (VolumeRelax(pressure=1.5), "volume_relax(pressure=1.5)"),
        (SymmetryRelax(pressure=10), "symmetry_relax(pressure=10)"),
        (FullRelax(pressure=3.0), "full_relax(pressure=3.0)"),
        (FullRelax(pressure=-0.25), "full_relax(pressure=-0.25)"),
    ],
)
def test_pressure_enters_the_name(settings, name):
    """Relaxing against a pressure gives a different structure, so it belongs in the step."""
    assert str(settings) == name


@pytest.mark.parametrize("settings", [Relax(), CellRelax(), VolumeRelax(), SymmetryRelax(), FullRelax()])
def test_zero_or_absent_pressure_stays_out_of_the_name(settings):
    assert "pressure" not in str(settings)


def test_relaxation_records_its_pressure(cu):
    assert step_of(VolumeRelax(max_steps=1, pressure=2.0).relax(cu)) == "volume_relax(pressure=2.0)"


def test_later_relaxation_replaces_the_earlier_step(cu):
    (volmin,) = relax([cu], VolumeRelax(max_steps=1), Morse())
    (allmin,) = relax([volmin], FullRelax(max_steps=1), Morse())
    assert step_of(volmin) == "volume_relax"
    assert step_of(allmin) == "full_relax"


def test_relax_does_not_tag_its_input(cu):
    FullRelax(max_steps=1).relax(cu)
    assert STEP_KEY not in cu.info


# --- perturbations ---


def test_perturbation_records_its_name(cu2):
    assert step_of(Rattle(0.1)(cu2)) == "rattle(0.1)"


def test_series_records_only_its_last_perturbation(cu2):
    """``step`` is the last thing done; the cumulative view stays in ``perturbation``."""
    perturbed = Series((Rattle(0.1), Stretch(hydro=0.1, shear=0.1)))(cu2)
    assert step_of(perturbed) == "stretch(hydro=0.1, shear=0.1)"
    assert perturbed.info["perturbation"] == "rattle(0.1)+stretch(hydro=0.1, shear=0.1)"


def test_perturbation_step_agrees_with_perturbation_key(cu2):
    perturbed = Rattle(0.1)(cu2)
    assert step_of(perturbed) == perturbed.info["perturbation"]


def test_perturbation_replaces_the_relaxation_step(cu):
    relaxed = FullRelax(max_steps=1).relax(cu)
    (perturbed,) = perturb([relaxed], [Rattle(0.05)])
    assert step_of(relaxed) == "full_relax"
    assert step_of(perturbed) == "rattle(0.05)"


def test_bare_perturbation_function_does_not_tag(cu2):
    """The inplace helpers are not workflow steps; only PerturbationABC records one."""
    rattle(cu2, 0.1)
    assert STEP_KEY not in cu2.info


# --- generation ---


def test_pyxtal_tags_generated_structures():
    assert step_of(pyxtal(1, species=["Cu"], num_ions=[2])) == "pyxtal"


def test_sample_tags_generated_structures():
    structures = list(sample(Formulas.range("Cu", 1, 3), spacegroups=[225, 194], max_atoms=2, rng=0))
    assert len(structures) > 0
    assert {step_of(s) for s in structures} == {"pyxtal"}


# --- the plain ASSYST workflow ---


def test_workflow_steps_are_distinguishable():
    """The three unperturbed sets of a plain run differ only in their step, so it has to tell them apart."""
    calculator = Morse()
    settings = {"max_steps": 3, "force_tolerance": 1e-2}

    spg = list(sample(Formulas.range("Cu", 1, 3), spacegroups=[225, 194], max_atoms=2, rng=0))
    volmin = list(relax(spg, VolumeRelax(**settings), calculator))
    allmin = list(relax(volmin, FullRelax(**settings), calculator))
    perturbed = list(
        perturb(
            allmin,
            [Rattle(0.05, create_supercells=True, rng=1), Stretch(hydro=0.1, shear=0.02, rng=2)],
        )
    )
    assert len(spg) > 0 and len(perturbed) > 0

    assert {step_of(s) for s in spg} == {"pyxtal"}
    assert {step_of(s) for s in volmin} == {"volume_relax"}
    assert {step_of(s) for s in allmin} == {"full_relax"}
    assert {step_of(s) for s in perturbed} == {"rattle(0.05)", "stretch(hydro=0.1, shear=0.02)"}

    # what the key is for: no two of the five sets share a step
    assert len({step_of(s) for s in spg + volmin + allmin + perturbed}) == 5
