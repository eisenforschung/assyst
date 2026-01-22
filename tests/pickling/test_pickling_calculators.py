import pickle
import pytest
from assyst.calculators import Morse, Grace

try:
    import tensorpotential
    HAS_TENSORPOTENTIAL = True
except ImportError:
    HAS_TENSORPOTENTIAL = False

def test_pickling_morse():
    m = Morse(epsilon=1.0, r0=1.0, rho0=1.0)
    p = pickle.dumps(m)
    m2 = pickle.loads(p)
    assert m2 == m

@pytest.mark.skipif(not HAS_TENSORPOTENTIAL, reason="tensorpotential not installed")
def test_pickling_grace():
    g = Grace(model='GRACE-FS-OAM')
    p = pickle.dumps(g)
    g2 = pickle.loads(p)
    assert g2 == g
