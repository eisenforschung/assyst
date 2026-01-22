import pickle
from assyst.crystals import Formulas

def test_pickling_formulas():
    f = Formulas.range('H', 1, 3)
    p = pickle.dumps(f)
    f2 = pickle.loads(p)
    assert f2 == f
