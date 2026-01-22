import pickle
from assyst.relax import Relax, CellRelax, VolumeRelax, SymmetryRelax, FullRelax

def test_pickling_relax_classes():
    for cls in [Relax, CellRelax, VolumeRelax, SymmetryRelax, FullRelax]:
        r = cls(max_steps=100, force_tolerance=1e-3)
        p = pickle.dumps(r)
        r2 = pickle.loads(p)
        assert r2 == r
