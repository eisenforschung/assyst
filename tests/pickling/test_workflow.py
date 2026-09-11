import pickle

from assyst.calculators import Morse
from assyst.filters import DistanceFilter
from assyst.perturbations import Rattle, Stretch
from assyst.relaxations import FullRelax, VolumeRelax
from assyst.workflow import FilterStage, PerturbStage, RelaxStage


def test_pickling_stages():
    stages = [
        RelaxStage(VolumeRelax(max_steps=3), Morse(epsilon=0.3, r0=2.5, rho0=4)),
        RelaxStage(FullRelax(max_steps=3), Morse()),
        PerturbStage(
            (Rattle(0.25, rng=0), Stretch(hydro=0.05, shear=0.005, rng=0)),
            filters=[DistanceFilter({"Cu": 1})],
        ),
        FilterStage(DistanceFilter({"Cu": 1})),
    ]

    for stage in stages:
        loaded = pickle.loads(pickle.dumps(stage))
        assert type(loaded) is type(stage)
