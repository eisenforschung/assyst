import pickle
from assyst.filters import AndFilter, OrFilter, DistanceFilter, AspectFilter, VolumeFilter, EnergyFilter, ForceFilter

def test_pickling_filters():
    df = DistanceFilter({'H': 1.0})
    vf = VolumeFilter(20.0)
    af = AspectFilter(6.0)
    ef = EnergyFilter(min_energy=-10.0, max_energy=0.0)
    ff = ForceFilter(max_force=1.0)
    and_f = AndFilter(df, vf)
    or_f = OrFilter(df, vf)

    for f in [df, vf, af, ef, ff, and_f, or_f]:
        p = pickle.dumps(f)
        f2 = pickle.loads(p)
        assert type(f2) is type(f)
        assert f2 == f
