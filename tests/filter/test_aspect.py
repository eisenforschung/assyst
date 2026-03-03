from ase import Atoms
from ase.cell import Cell
from assyst.filters import AspectFilter
from hypothesis import given, strategies as st
from pyxtal.lattice import generate_cellpara
from tests.strategies.strategies import cells


@given(cells(), st.floats(0, exclude_min=True))
def test_aspect_filter(cell, ratio):
    filter = AspectFilter(maximum_aspect_ratio=ratio)

    structure = Atoms('Cu', cell=cell, pbc=True)
    c_over_a = max(structure.cell.lengths()) / min(structure.cell.lengths())
    assert filter(structure) == (c_over_a <= ratio), \
        "AspectFilter should filter only structures larger than given aspect ratio!"
    filter = AspectFilter(maximum_aspect_ratio=c_over_a)
    assert filter(structure), \
        "AspectFilter should not filter structures with exactly the given aspect ratio!"
