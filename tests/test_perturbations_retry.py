from unittest.mock import Mock
from ase import Atoms
from assyst.perturbations import apply_perturbations

def test_filter_retry_success():
    """Test that apply_perturbations retries and eventually succeeds when filter fails."""
    structure = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    # Filter fails 3 times, then succeeds.
    ff = Mock(side_effect=[False, False, False, True])
    identity = lambda s: s
    results = list(apply_perturbations([structure], [identity], filters=[ff], retries=10))
    assert len(results) == 1
    assert ff.call_count == 4

def test_filter_retry_failure():
    """Test that apply_perturbations gives up after retries limit when filter keeps failing."""
    structure = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    # Filter always fails
    ff = Mock(return_value=False)
    identity = lambda s: s
    results = list(apply_perturbations([structure], [identity], filters=[ff], retries=10))
    assert len(results) == 0
    assert ff.call_count == 10

def test_value_error_no_retry():
    """Test that apply_perturbations does NOT retry and silently ignores when ValueError is raised."""
    structure = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    fp = Mock(side_effect=ValueError("Intentional ValueError"))

    results = list(apply_perturbations([structure], [fp], retries=10))
    assert len(results) == 0
    assert fp.call_count == 1

def test_mixed_no_retry_on_value_error():
    """Test that ValueError stops retries even if some attempts succeeded in reaching filters."""
    structure = Atoms('H2', positions=[[0, 0, 0], [0.74, 0, 0]], cell=[10, 10, 10])
    # First call returns structure, second raises ValueError
    fp = Mock(side_effect=[structure, ValueError("Intentional ValueError")])
    # Filter always fails
    ff = Mock(return_value=False)

    results = list(apply_perturbations([structure], [fp], filters=[ff], retries=10))
    assert len(results) == 0
    assert fp.call_count == 2
