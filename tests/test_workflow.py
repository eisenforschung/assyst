"""Tests for the full-workflow drivers in :mod:`assyst.workflow`."""

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from assyst.calculators import Morse
from assyst.crystals import Formulas, sample
from assyst.filters import AspectFilter, DistanceFilter
from assyst.perturbations import Rattle, Stretch
from assyst.relaxations import FullRelax, VolumeRelax
from assyst.workflow import (
    PerturbStage,
    RelaxStage,
    breadth_first,
    depth_first,
)


# --- deterministic synthetic stages, so set membership can be checked exactly ---


@dataclass(frozen=True)
class Tag:
    """Synthetic stage: emit a child per structure with a deterministic uuid.

    Defined at module level (and picklable) so it can be shipped to a process
    based executor in the depth-first tests.
    """

    tag: str

    def __call__(self, structures):
        for s in structures:
            child = s.copy()
            child.info["uuid"] = s.info["uuid"] + self.tag
            yield child


@dataclass(frozen=True)
class KeepWithTag:
    """Synthetic filter stage keeping only structures whose uuid contains `needle`."""

    needle: str

    def __call__(self, structures):
        return (s for s in structures if self.needle in s.info["uuid"])


class SerialExecutor:
    """Minimal executor stub exposing the ``map`` seam depth_first relies on."""

    def map(self, fn, iterable):
        return [fn(x) for x in iterable]


def make_seeds(names):
    seeds = []
    for name in names:
        a = Atoms("Cu", positions=[[0, 0, 0]], cell=np.eye(3) * 4, pbc=True)
        a.info["uuid"] = name
        seeds.append(a)
    return seeds


def uuids(structures):
    return [s.info["uuid"] for s in structures]


# --------------------------------------------------------------------------- #
# Driver semantics on synthetic, fully deterministic stages
# --------------------------------------------------------------------------- #


def test_breadth_first_collects_seeds_and_every_stage():
    seeds = make_seeds(["a", "b"])
    result = breadth_first(seeds, [Tag("/1"), Tag("/2")])
    # seeds + first stage + second stage, all distinct
    assert uuids(result) == ["a", "b", "a/1", "b/1", "a/1/2", "b/1/2"]


def test_breadth_first_no_stages_returns_seeds():
    seeds = make_seeds(["a", "b"])
    assert uuids(breadth_first(seeds, [])) == ["a", "b"]


def test_filter_stage_narrows_flow_without_duplicating():
    seeds = make_seeds(["a", "b", "c"])
    # keep only 'a' and 'c', then tag the survivors
    result = breadth_first(seeds, [KeepWithTag("a"), KeepWithTag("c")])
    # the filtered structures keep their uuid, so they are deduplicated away;
    # only 'a' survives both filters and it is already among the seeds
    assert uuids(result) == ["a", "b", "c"]

    result = breadth_first(seeds, [KeepWithTag("a"), Tag("/x")])
    # 'a' passes the filter and is then tagged; 'b'/'c' never flow onward
    assert uuids(result) == ["a", "b", "c", "a/x"]


def test_depth_first_matches_breadth_first_set():
    seeds = make_seeds(["a", "b", "c"])
    stages = [Tag("/1"), Tag("/2")]
    bf = breadth_first(seeds, stages)
    df = depth_first(seeds, stages)
    # identical data set, only the ordering differs (grouped by seed)
    assert set(uuids(df)) == set(uuids(bf))
    assert len(df) == len(bf)
    # depth-first groups every seed's subtree together
    assert uuids(df) == ["a", "a/1", "a/1/2", "b", "b/1", "b/1/2", "c", "c/1", "c/1/2"]


def test_depth_first_accepts_injected_executor():
    seeds = make_seeds(["a", "b", "c"])
    stages = [Tag("/1"), Tag("/2")]
    serial = depth_first(seeds, stages)
    via_executor = depth_first(seeds, stages, executor=SerialExecutor())
    assert uuids(via_executor) == uuids(serial)


def test_depth_first_with_process_pool():
    # Exercises the real parallel path: stages must survive pickling and run
    # in a separate process.
    from concurrent.futures import ProcessPoolExecutor

    seeds = make_seeds(["a", "b", "c"])
    stages = [Tag("/1"), Tag("/2")]
    expected = set(uuids(breadth_first(seeds, stages)))
    with ProcessPoolExecutor(max_workers=2) as executor:
        result = depth_first(seeds, stages, executor=executor)
    assert set(uuids(result)) == expected


# --------------------------------------------------------------------------- #
# Integration: a real sample -> relax -> relax -> perturb pipeline
# --------------------------------------------------------------------------- #


def make_pipeline_stages():
    reference = Morse(epsilon=0.3, r0=2.55265548 * 1.10619396, rho0=4)
    return [
        RelaxStage(VolumeRelax(max_steps=3, force_tolerance=1e-2), reference),
        RelaxStage(FullRelax(max_steps=3, force_tolerance=1e-2), reference),
        PerturbStage(
            (Rattle(0.25, rng=0), Stretch(hydro=0.05, shear=0.005, rng=0)),
            filters=[DistanceFilter({"Cu": 1})],
        ),
    ]


def test_real_pipeline_breadth_first_runs():
    rng = np.random.default_rng(0)
    seeds = list(
        filter(
            AspectFilter(6),
            sample(Formulas.range("Cu", 3), spacegroups=[225, 194], max_atoms=2, rng=rng),
        )
    )
    assert len(seeds) > 0

    result = breadth_first(seeds, make_pipeline_stages())

    # every seed is part of the full data set, and everything is a real,
    # uuid-tracked structure
    seed_ids = set(uuids(seeds))
    result_ids = set(uuids(result))
    assert seed_ids <= result_ids
    assert all(isinstance(s, Atoms) for s in result)
    # relaxation alone produces a relaxed copy of each seed in both stages
    assert len(result) > len(seeds)


def test_real_pipeline_depth_first_same_dataset_as_breadth_first():
    rng = np.random.default_rng(0)
    seeds = list(
        filter(
            AspectFilter(6),
            sample(Formulas.range("Cu", 3), spacegroups=[225, 194], max_atoms=2, rng=rng),
        )
    )
    assert len(seeds) > 0

    # Fresh stages per driver so the perturbation RNGs start from the same
    # state; serial depth-first consumes the RNG in the same order as
    # breadth-first, so the generated structures are identical.
    bf = breadth_first(seeds, make_pipeline_stages())
    df = depth_first(seeds, make_pipeline_stages())

    def fingerprint(s):
        return (
            s.get_chemical_formula(),
            tuple(np.round(s.positions.flatten(), 5)),
            tuple(np.round(s.cell.array.flatten(), 5)),
        )

    assert len(df) == len(bf)
    assert sorted(map(fingerprint, df)) == sorted(map(fingerprint, bf))
