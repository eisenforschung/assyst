import pickle
from hypothesis import given, strategies as st
from assyst.crystals import Formulas

@given(st.dictionaries(st.text(min_size=1, max_size=2), st.integers(min_value=1, max_value=10), min_size=1, max_size=3))
def test_pickling_formulas(atoms_dict):
    f = Formulas((atoms_dict,))
    p = pickle.dumps(f)
    f2 = pickle.loads(p)
    assert f2 == f
