from ase import Atoms
from ase.cell import Cell
from assyst.filters import VolumeFilter

# TODO unify duplicated code
from hypothesis import given, strategies as st
from pyxtal.lattice import generate_cellpara


@st.composite
def cells(draw):
    ltype = st.sampled_from(["monoclinic", "triclinic", "orthorhombic", "tetragonal", "hexagonal", "trigonal", "cubic"])
    volume = st.floats(min_value=5, max_value=100, allow_nan=False, allow_infinity=False)
    return Cell.fromcellpar(generate_cellpara(draw(ltype), draw(volume)))


@given(cells(), st.floats(0, exclude_min=True))
def test_volume_filter(cell, maximum_volume_per_atom):
    filter = VolumeFilter(maximum_volume_per_atom)

    structure = Atoms('Cu', cell=cell, pbc=True)
    volume = structure.cell.volume/len(structure)
    assert filter(structure) == (volume <= maximum_volume_per_atom), \
        "VolumeFilter should filter only structures larger than given volume!"
    filter = VolumeFilter(volume)
    assert filter(structure), \
        "VolumeFilter should not filter structures with exactly the given volume!"
