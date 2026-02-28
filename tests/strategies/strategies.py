from hypothesis import strategies as st
from pyxtal.lattice import generate_cellpara
from ase.cell import Cell
from ase.data import atomic_numbers
from ase.build import bulk

@st.composite
def cells(draw):
    ltype = st.sampled_from(["monoclinic", "triclinic", "orthorhombic", "tetragonal", "hexagonal", "trigonal", "cubic"])
    volume = st.floats(min_value=5, max_value=100, allow_nan=False, allow_infinity=False)
    return Cell.fromcellpar(generate_cellpara(draw(ltype), draw(volume)))

def elements():
    # pyxtal tol somehow only supports until element 105
    return st.sampled_from(list(atomic_numbers.keys())[1:106])

@st.composite
def random_element_structures(draw):
    """Return structures with random elements inside"""
    structure = bulk("Cu", cubic=True).repeat(3)
    elements_list = st.lists(elements(),
                             min_size=len(structure), max_size=len(structure))
    structure.symbols[:] = draw(elements_list)
    return structure
