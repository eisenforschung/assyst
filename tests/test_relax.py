from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter

from assyst.calculators import AseCalculatorConfig, Morse
from assyst.relaxations import CellRelax, FullRelax, Relax, SymmetryRelax, VolumeRelax, relax


class MockCalculator:
    def get_potential_energy(self, atoms=None):
        return 0.0

    def get_forces(self, atoms=None):
        return [[0.0, 0.0, 0.0]]

    def get_stress(self, atoms=None):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def single_atom_structure():
    s = Atoms("H", positions=[[0, 0, 0]], cell=[10, 10, 10], pbc=True)
    s.calc = MockCalculator()
    return s


@pytest.fixture
def cu_structure():
    s = bulk("Cu", cubic=True)
    s.calc = Morse().get_calculator()
    return s


@pytest.fixture
def cu_structures():
    structures = [bulk("Cu", cubic=True) for _ in range(3)]
    for s in structures:
        s.positions[0] += np.random.uniform(-0.1, 0.1, size=3)
    return structures


# --- Basic relax() method tests ---


@patch("ase.optimize.LBFGS.run")
def test_relax_runs(mock_run, single_atom_structure):
    relaxed = Relax().relax(single_atom_structure)
    mock_run.assert_called_once()
    assert isinstance(relaxed.calc, SinglePointCalculator)


@patch("ase.optimize.BFGS.run")
def test_relax_bfgs(mock_run, single_atom_structure):
    relaxed = Relax(algorithm="BFGS").relax(single_atom_structure)
    mock_run.assert_called_once()
    assert isinstance(relaxed.calc, SinglePointCalculator)


@patch("ase.optimize.FIRE.run")
def test_relax_fire(mock_run, single_atom_structure):
    relaxed = Relax(algorithm="FIRE").relax(single_atom_structure)
    mock_run.assert_called_once()
    assert isinstance(relaxed.calc, SinglePointCalculator)


@patch("ase.optimize.LBFGS.run")
def test_cell_relax_runs(mock_run, single_atom_structure):
    CellRelax().relax(single_atom_structure)
    mock_run.assert_called_once()


@patch("ase.optimize.LBFGS.run")
def test_volume_relax_runs(mock_run, single_atom_structure):
    VolumeRelax(pressure=1.0).relax(single_atom_structure)
    mock_run.assert_called_once()


@patch("ase.optimize.LBFGS.run")
def test_symmetry_relax_runs(mock_run, single_atom_structure):
    SymmetryRelax().relax(single_atom_structure)
    mock_run.assert_called_once()


@patch("ase.optimize.LBFGS.run")
def test_full_relax_runs(mock_run, single_atom_structure):
    FullRelax(pressure=1.0).relax(single_atom_structure)
    mock_run.assert_called_once()


@patch("assyst.relaxations.Relax.relax")
def test_relax_function_with_calc_object(mock_relax_method, single_atom_structure):
    calculator = MockCalculator()
    list(relax([single_atom_structure], Relax(), calculator))
    assert mock_relax_method.call_args[0][0].calc is calculator


@patch("assyst.relaxations.Relax.relax")
def test_relax_function_with_calc_config(mock_relax_method, single_atom_structure):
    class MockCalcConfig(AseCalculatorConfig):
        def get_calculator(self):
            return MockCalculator()

    list(relax([single_atom_structure], Relax(), MockCalcConfig()))
    assert isinstance(mock_relax_method.call_args[0][0].calc, MockCalculator)


# --- apply_filter_and_constraints tests ---


def test_base_relax_returns_structure_unchanged(cu_structure):
    result = Relax().apply_filter_and_constraints(cu_structure)
    assert result is cu_structure
    assert len(cu_structure.constraints) == 0


def test_cell_relax_returns_frechet_cell_filter(cu_structure):
    result = CellRelax().apply_filter_and_constraints(cu_structure)
    assert isinstance(result, FrechetCellFilter)
    assert result.constant_volume


def test_cell_relax_adds_fix_atoms_constraint(cu_structure):
    CellRelax().apply_filter_and_constraints(cu_structure)
    assert len(cu_structure.constraints) == 1
    assert isinstance(cu_structure.constraints[0], FixAtoms)


def test_volume_relax_returns_frechet_cell_filter_with_hydrostatic(cu_structure):
    result = VolumeRelax(pressure=2.0).apply_filter_and_constraints(cu_structure)
    assert isinstance(result, FrechetCellFilter)
    assert result.hydrostatic_strain
    assert result.scalar_pressure == 2.0


def test_volume_relax_adds_fix_atoms_constraint(cu_structure):
    VolumeRelax().apply_filter_and_constraints(cu_structure)
    assert len(cu_structure.constraints) == 1
    assert isinstance(cu_structure.constraints[0], FixAtoms)


def test_volume_relax_default_pressure_is_zero(cu_structure):
    result = VolumeRelax().apply_filter_and_constraints(cu_structure)
    assert result.scalar_pressure == 0.0


def test_symmetry_relax_returns_frechet_cell_filter(cu_structure):
    result = SymmetryRelax(pressure=0.5).apply_filter_and_constraints(cu_structure)
    assert isinstance(result, FrechetCellFilter)
    assert result.scalar_pressure == 0.5


def test_symmetry_relax_adds_fix_symmetry_constraint(cu_structure):
    SymmetryRelax().apply_filter_and_constraints(cu_structure)
    assert len(cu_structure.constraints) == 1
    assert isinstance(cu_structure.constraints[0], FixSymmetry)


def test_full_relax_returns_frechet_cell_filter(cu_structure):
    result = FullRelax(pressure=1.5).apply_filter_and_constraints(cu_structure)
    assert isinstance(result, FrechetCellFilter)
    assert result.scalar_pressure == 1.5


def test_full_relax_does_not_add_constraints(cu_structure):
    FullRelax().apply_filter_and_constraints(cu_structure)
    assert len(cu_structure.constraints) == 0


# --- Integration tests with real Morse calculator ---


def test_relax_returns_single_point_calculator_with_valid_energy(cu_structure):
    result = Relax(max_steps=50).relax(cu_structure)
    assert isinstance(result.calc, SinglePointCalculator)
    energy = result.calc.get_potential_energy()
    assert isinstance(energy, float)
    assert not np.isnan(energy)


def test_relax_returns_forces_with_correct_shape(cu_structure):
    result = Relax(max_steps=50).relax(cu_structure)
    assert result.calc.get_forces().shape == (len(cu_structure), 3)


def test_relax_returns_stress_with_correct_shape(cu_structure):
    result = Relax(max_steps=50).relax(cu_structure)
    assert result.calc.get_stress().shape == (6,)


def test_relax_clears_constraints_on_output(cu_structure):
    result = CellRelax(max_steps=10).relax(cu_structure)
    assert len(result.constraints) == 0


def test_relax_does_not_modify_input_structure_positions(cu_structure):
    original_positions = cu_structure.get_positions().copy()
    Relax(max_steps=5).relax(cu_structure)
    np.testing.assert_array_equal(cu_structure.get_positions(), original_positions)


def test_relax_assigns_new_uuid(cu_structure):
    cu_structure.info["uuid"] = "original-uuid"
    result = Relax(max_steps=5).relax(cu_structure)
    assert "uuid" in result.info
    assert result.info["uuid"] != "original-uuid"


def test_relax_tracks_uuid_lineage(cu_structure):
    cu_structure.info["uuid"] = "parent-uuid"
    result = Relax(max_steps=5).relax(cu_structure)
    assert "lineage" in result.info
    assert "parent-uuid" in result.info["lineage"]


def test_relax_preserves_seed(cu_structure):
    cu_structure.info["uuid"] = "first-uuid"
    cu_structure.info["seed"] = "seed-value"
    result = Relax(max_steps=5).relax(cu_structure)
    assert result.info["seed"] == "seed-value"


def test_relax_assigns_seed_if_absent(cu_structure):
    result = Relax(max_steps=5).relax(cu_structure)
    assert "seed" in result.info
    assert result.info["seed"] == result.info["uuid"]


def test_relax_reduces_energy():
    s = bulk("Cu", cubic=True)
    s.positions[0] += 0.3
    s.calc = Morse().get_calculator()
    initial_energy = s.get_potential_energy()
    result = Relax(max_steps=100).relax(s)
    assert result.calc.get_potential_energy() < initial_energy


def test_full_relax_converges(cu_structure):
    result = FullRelax(max_steps=200).relax(cu_structure)
    assert np.all(np.abs(result.calc.get_forces()) < 1e-2)


# --- Dataclass equality and hash tests ---


def test_relax_equal_with_same_params():
    assert Relax(max_steps=50) == Relax(max_steps=50)


def test_relax_not_equal_with_different_params():
    assert Relax(max_steps=50) != Relax(max_steps=100)


def test_relax_hashable():
    r = Relax()
    assert hash(r) == hash(r)
    assert r in {r}


def test_cell_relax_equal_with_same_params():
    assert CellRelax(max_steps=50) == CellRelax(max_steps=50)


def test_volume_relax_equal_with_same_pressure():
    assert VolumeRelax(pressure=1.0) == VolumeRelax(pressure=1.0)


def test_volume_relax_not_equal_with_different_pressure():
    assert VolumeRelax(pressure=0.0) != VolumeRelax(pressure=1.0)


def test_symmetry_relax_equal_with_same_params():
    assert SymmetryRelax(pressure=0.5) == SymmetryRelax(pressure=0.5)


def test_full_relax_equal_with_same_params():
    assert FullRelax(pressure=0.0) == FullRelax(pressure=0.0)


def test_different_relax_types_not_equal():
    assert Relax() != CellRelax()
    assert CellRelax() != VolumeRelax()
    assert SymmetryRelax() != FullRelax()


def test_relax_default_values():
    r = Relax()
    assert r.max_steps == 100
    assert r.force_tolerance == pytest.approx(1e-3)
    assert r.algorithm == "LBFGS"


def test_volume_relax_default_pressure():
    assert VolumeRelax().pressure == 0.0


def test_symmetry_relax_default_pressure():
    assert SymmetryRelax().pressure == 0.0


def test_full_relax_default_pressure():
    assert FullRelax().pressure == 0.0


# --- relax() generator function tests ---


def test_relax_function_yields_correct_number_of_structures(cu_structures):
    results = list(relax(cu_structures, Relax(max_steps=5), Morse()))
    assert len(results) == len(cu_structures)


def test_relax_function_does_not_mutate_input(cu_structures):
    original_positions = [s.get_positions().copy() for s in cu_structures]
    list(relax(cu_structures, Relax(max_steps=5), Morse()))
    for s, orig in zip(cu_structures, original_positions):
        np.testing.assert_array_equal(s.get_positions(), orig)


def test_relax_function_attaches_single_point_calculator(cu_structures):
    for result in relax(cu_structures, Relax(max_steps=5), Morse()):
        assert isinstance(result.calc, SinglePointCalculator)


def test_relax_function_with_ase_calculator_config(cu_structures):
    results = list(relax(cu_structures, Relax(max_steps=5), Morse()))
    assert len(results) == 3


def test_relax_function_with_raw_ase_calculator(cu_structures):
    calc = Morse().get_calculator()
    results = list(relax(cu_structures, Relax(max_steps=5), calc))
    assert len(results) == 3


def test_relax_function_is_lazy_generator(cu_structures):
    call_count = [0]
    original_relax = Relax.relax

    def counting_relax(self_inner, structure):
        call_count[0] += 1
        return original_relax(self_inner, structure)

    with patch.object(Relax, "relax", counting_relax):
        gen = relax(cu_structures, Relax(max_steps=5), Morse())
        assert call_count[0] == 0, "relax should not be called before iteration"
        next(gen)
        assert call_count[0] == 1
