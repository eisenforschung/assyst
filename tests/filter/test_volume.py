from ase import Atoms
from ase.cell import Cell
from assyst.filters import VolumeFilter

from hypothesis import given, strategies as st
from pyxtal.lattice import generate_cellpara
from tests.strategies.strategies import cells


@given(cells(), st.floats(0, exclude_min=True))
def test_volume_filter_maximum(cell, maximum_volume_per_atom):
    filter = VolumeFilter(maximum_volume_per_atom)

    structure = Atoms('Cu', cell=cell, pbc=True)
    volume = structure.cell.volume/len(structure)
    assert filter(structure) == (volume <= maximum_volume_per_atom), \
        "VolumeFilter should filter only structures larger than given volume!"
    filter = VolumeFilter(volume)
    assert filter(structure), \
        "VolumeFilter should not filter structures with exactly the given maximum volume!"


@given(cells(), st.floats(0, exclude_min=True))
def test_volume_filter_minimum(cell, minimum_volume_per_atom):
    filter = VolumeFilter(minimum_volume_per_atom=minimum_volume_per_atom)

    structure = Atoms('Cu', cell=cell, pbc=True)
    volume = structure.cell.volume/len(structure)
    assert filter(structure) == (volume >= minimum_volume_per_atom), \
        "VolumeFilter should filter only structures smaller than given volume!"
    filter = VolumeFilter(minimum_volume_per_atom=volume)
    assert filter(structure), \
        "VolumeFilter should not filter structures with exactly the given minimum volume!"


@given(cells(), st.floats(0, exclude_min=True), st.floats(0, exclude_min=True))
def test_volume_filter_range(cell, a, b):
    minimum, maximum = sorted((a, b))
    filter = VolumeFilter(minimum, maximum)

    structure = Atoms('Cu', cell=cell, pbc=True)
    volume = structure.cell.volume/len(structure)
    assert filter(structure) == (minimum <= volume <= maximum), \
        "VolumeFilter should filter only structures outside given volume range!"
