import pytest
from ase import Atoms
from assyst.crystals import pyxtal


def test_num_ions_and_species_length_mismatch_one():
    """ValueError must be raised if num_ions and species lengths should not match"""
    with pytest.raises(ValueError):
        pyxtal(1, species=["Fe"], num_ions=[1, 2])

    with pytest.raises(ValueError):
        pyxtal(1, species=["Fe", "Cr"], num_ions=[1])


def test_allow_exceptions_true_warns():
    """allow_exceptions=True should suppress the error and emit a warning!"""
    try:
        with pytest.warns(
            UserWarning,
            match=r"Groups \[193, 194\] could not be generated with stoichiometry Mg1!",
        ) as record:
            pyxtal([193, 194], ["Mg"], num_ions=[1], allow_exceptions=True)

        assert len(record) == 1, "warning count should be exactly 1!"
    except ValueError:
        pytest.fail("Error should not be raised when allow_exceptions=True is passed!")


def test_allow_exceptions_false_raises():
    """allow_exceptions=False should let the ValueError propagate!"""
    with pytest.raises(ValueError):
        pyxtal(194, ["Mg"], num_ions=[1], allow_exceptions=False)


def test_return_is_atoms_instance():
    """pyxtal should return an Atoms instance for scalar arguments!"""
    result = pyxtal(1, species=["Fe"], num_ions=[1])
    assert isinstance(result, Atoms), "result should be an Atoms instance!"


def test_return_is_list_for_multiple_groups():
    """pyxtal should return a list when multiple space‑group numbers are supplied!"""
    result = pyxtal([1, 2], species=["Fe"], num_ions=[1])
    assert isinstance(result, list), "result should be a list when multiple groups are supplied!"


def test_return_is_list_when_repeat_given():
    """pyxtal should return a list when the repeat argument is used!"""
    result = pyxtal(1, species=["Fe"], num_ions=[1], repeat=5)
    assert isinstance(result, list), "result should be a list when repeat is given!"


def test_repeat_length_matches_requested():
    """pyxtal should produce exactly the requested number of structures when repeat is set!"""
    result = pyxtal(1, species=["Fe"], num_ions=[1], repeat=5)
    assert len(result) == 5, "length of result should match the repeat value!"


def test_repeat_elements_are_dicts():
    """pyxtal should return a list of dicts when repeat is used!"""
    result = pyxtal(1, species=["Fe"], num_ions=[1], repeat=5)
    assert all(isinstance(d, dict) for d in result), "every element in result should be a dict!"
