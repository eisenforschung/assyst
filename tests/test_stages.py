"""The ``stage`` key records which workflow steps produced a structure, in the order they ran."""

import pytest
from ase import Atoms
from ase.build import bulk

from assyst.calculators import Morse
from assyst.crystals import Formulas, pyxtal, sample
from assyst.perturbations import Rattle, Series, Stretch, perturb, rattle
from assyst.relaxations import CellRelax, FullRelax, Relax, SymmetryRelax, VolumeRelax, relax
from assyst.utils import STAGE_KEY, record_stage, stage_of


@pytest.fixture
def cu():
    s = bulk("Cu", cubic=True)
    s.calc = Morse().get_calculator()
    return s


@pytest.fixture
def cu2():
    return Atoms("Cu2", positions=[[0, 0, 0], [1.5, 1.5, 1.5]], cell=[3, 3, 3], pbc=True)


# --- record_stage / stage_of ---


def test_record_stage_sets_key():
    s = Atoms("H")
    record_stage(s, "spg")
    assert s.info[STAGE_KEY] == "spg"


def test_record_stage_appends_in_order():
    s = Atoms("H")
    for step in ("spg", "volume_relax", "full_relax"):
        record_stage(s, step)
    assert s.info[STAGE_KEY] == "spg+volume_relax+full_relax"


def test_record_stage_operates_inplace():
    s = Atoms("H")
    assert record_stage(s, "spg") is s


def test_record_stage_repeats_the_same_step():
    """Relaxing twice with the same settings is two steps, not one."""
    s = Atoms("H")
    record_stage(s, "full_relax")
    record_stage(s, "full_relax")
    assert s.info[STAGE_KEY] == "full_relax+full_relax"


def test_stage_of_reads_the_key():
    s = Atoms("H")
    s.info[STAGE_KEY] = "spg+volume_relax"
    assert stage_of(s) == "spg+volume_relax"


def test_stage_of_defaults_for_foreign_structures():
    assert stage_of(Atoms("H")) == "unknown"
    assert stage_of(Atoms("H"), default="external") == "external"


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
    assert stage_of(settings(max_steps=1).relax(cu)) == name


def test_relaxation_name_ignores_settings():
    """The kind of relaxation is the stage, its numerics are not."""
    assert str(FullRelax(max_steps=5, force_tolerance=1e-1, pressure=3.0)) == str(FullRelax())


def test_relaxations_accumulate(cu):
    (volmin,) = relax([cu], VolumeRelax(max_steps=1), Morse())
    (allmin,) = relax([volmin], FullRelax(max_steps=1), Morse())
    assert stage_of(volmin) == "volume_relax"
    assert stage_of(allmin) == "volume_relax+full_relax"


def test_relax_generator_records(cu):
    (relaxed,) = relax([cu], VolumeRelax(max_steps=1), Morse())
    assert stage_of(relaxed) == "volume_relax"


def test_relax_does_not_tag_its_input(cu):
    FullRelax(max_steps=1).relax(cu)
    assert STAGE_KEY not in cu.info


# --- perturbations ---


def test_perturbation_records_its_name(cu2):
    assert stage_of(Rattle(0.1)(cu2)) == "rattle(0.1)"


def test_series_records_every_perturbation(cu2):
    perturbed = Series((Rattle(0.1), Stretch(hydro=0.1, shear=0.1)))(cu2)
    assert stage_of(perturbed) == "rattle(0.1)+stretch(hydro=0.1, shear=0.1)"


def test_perturbation_stage_agrees_with_perturbation_key(cu2):
    perturbed = (Rattle(0.1) + Stretch(hydro=0.1, shear=0.1))(cu2)
    assert stage_of(perturbed) == perturbed.info["perturbation"]


def test_perturbation_extends_the_relaxation_history(cu):
    relaxed = FullRelax(max_steps=1).relax(cu)
    (perturbed,) = perturb([relaxed], [Rattle(0.05)])
    assert stage_of(perturbed) == "full_relax+rattle(0.05)"


def test_bare_perturbation_function_does_not_tag(cu2):
    """The inplace helpers are not workflow steps; only PerturbationABC records one."""
    rattle(cu2, 0.1)
    assert STAGE_KEY not in cu2.info


# --- generation ---


def test_pyxtal_tags_generated_structures():
    assert stage_of(pyxtal(1, species=["Cu"], num_ions=[2])) == "spg"


def test_sample_tags_generated_structures():
    structures = list(sample(Formulas.range("Cu", 1, 3), spacegroups=[225, 194], max_atoms=2, rng=0))
    assert len(structures) > 0
    assert {stage_of(s) for s in structures} == {"spg"}


# --- the plain ASSYST workflow ---


def test_workflow_stages_are_distinguishable():
    """The three unperturbed sets of a plain run differ only in their stage, so it has to tell them apart."""
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

    assert {stage_of(s) for s in spg} == {"spg"}
    assert {stage_of(s) for s in volmin} == {"spg+volume_relax"}
    assert {stage_of(s) for s in allmin} == {"spg+volume_relax+full_relax"}
    assert {stage_of(s) for s in perturbed} == {
        "spg+volume_relax+full_relax+rattle(0.05)",
        "spg+volume_relax+full_relax+stretch(hydro=0.1, shear=0.02)",
    }

    # what the key is for: no two of the five sets share a stage
    assert len({stage_of(s) for s in spg + volmin + allmin + perturbed}) == 5
