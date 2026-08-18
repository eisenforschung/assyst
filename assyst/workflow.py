"""Convenience drivers that run a full ASSYST pipeline in one call.

The individual stages of ASSYST (:func:`~assyst.relaxations.relax`,
:func:`~assyst.perturbations.perturb`, filtering, …) all share the same
shape: they consume an :class:`collections.abc.Iterable` of
:class:`ase.Atoms` and yield another iterable of :class:`ase.Atoms`.  That
makes them trivially composable, but the canonical workflow still has to be
wired up by hand (see ``tests/reproducibility/test_full_workflow.py``).

This module captures that wiring behind two drivers that take the *seed*
structures plus an ordered list of :class:`Stage` objects:

* :func:`breadth_first` runs each stage over the whole batch before moving
  on to the next, returning every structure produced along the way -- the
  full data set in one call.
* :func:`depth_first` pushes each seed through the whole pipeline
  independently, which makes the per-seed subtrees embarrassingly parallel.
  An :class:`~concurrent.futures.Executor` (or anything exposing a
  compatible ``map``) can be injected to dispatch them, e.g. across an HPC
  allocation via `executorlib <https://github.com/pyiron/executorlib>`_.

Both drivers return the same *set* of structures (deduplicated by their
``uuid``); they only differ in the order in which the work is done and,
hence, in the order of the returned list.

A :class:`Stage` is simply a callable ``Iterable[Atoms] -> Iterable[Atoms]``.
Any such callable works; the :class:`RelaxStage`, :class:`PerturbStage` and
:class:`FilterStage` wrappers below bind a stage's configuration into a
small frozen dataclass so that the resulting stage stays hashable and
picklable -- a prerequisite for shipping it to remote workers.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Iterator

from ase import Atoms

from .calculators import AseCalculatorConfig
from .filters import Filter
from .perturbations import Perturbation, perturb
from .relaxations import Relax, relax

from ase.calculators.calculator import Calculator


Stage = Callable[[Iterable[Atoms]], Iterable[Atoms]]
"""A single workflow step: maps an iterable of structures to another one."""


@dataclass(frozen=True)
class RelaxStage:
    """A :func:`~assyst.relaxations.relax` call with its configuration bound.

    Args:
        settings (:class:`~assyst.relaxations.Relax`): the kind of relaxation to perform
        calculator (:class:`~assyst.calculators.AseCalculatorConfig` or :class:`ase.calculators.calculator.Calculator`): the energy/force engine to use
    """

    settings: Relax
    calculator: AseCalculatorConfig | Calculator

    def __call__(self, structures: Iterable[Atoms]) -> Iterator[Atoms]:
        return relax(structures, self.settings, self.calculator)


@dataclass(frozen=True)
class PerturbStage:
    """A :func:`~assyst.perturbations.perturb` call with its configuration bound.

    Args:
        perturbations (:class:`tuple` of :class:`~assyst.perturbations.Perturbation`): perturbations to apply to each structure
        filters (:class:`~assyst.filters.Filter` or iterable thereof, optional): filters every perturbed structure must pass
        retries (:class:`int`): max attempts per perturbation (default: 10)
    """

    perturbations: tuple[Perturbation, ...]
    filters: Iterable[Filter] | Filter | None = None
    retries: int = 10

    def __call__(self, structures: Iterable[Atoms]) -> Iterator[Atoms]:
        return perturb(
            structures,
            self.perturbations,
            filters=self.filters,
            retries=self.retries,
        )


@dataclass(frozen=True)
class FilterStage:
    """Drop structures that do not pass `filter` from the stream.

    Filtering does not create new structures, so a :class:`FilterStage`
    never adds anything to the data set collected by the drivers; it only
    narrows what flows on to the following stages.

    Args:
        filter (:class:`~assyst.filters.Filter`): predicate a structure must satisfy to be kept
    """

    filter: Filter

    def __call__(self, structures: Iterable[Atoms]) -> Iterator[Atoms]:
        return filter(self.filter, structures)


def _key(structure: Atoms):
    """Stable identity for deduplication: the uuid if present, else object id."""
    return structure.info.get("uuid", id(structure))


def _dedup(structures: Iterable[Atoms]) -> list[Atoms]:
    """Return the structures with duplicate uuids removed, keeping first occurrence."""
    seen = set()
    out = []
    for s in structures:
        k = _key(s)
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def breadth_first(
    structures: Iterable[Atoms],
    stages: Iterable[Stage],
) -> list[Atoms]:
    """Run every stage over the whole batch and collect all structures produced.

    The seed `structures` are passed through the first stage, whose output
    is passed through the second, and so on.  The returned list is the union
    of the seeds and the output of *every* stage, i.e. the full data set of
    the workflow in one call.  Structures sharing a ``uuid`` (for example the
    ones a :class:`FilterStage` lets through unchanged) appear only once.

    Args:
        structures (:class:`collections.abc.Iterable` of :class:`ase.Atoms`): the seed structures, e.g. from :func:`~assyst.crystals.sample`
        stages (:class:`collections.abc.Iterable` of :class:`Stage`): the workflow steps to run, in order

    Returns:
        :class:`list` of :class:`ase.Atoms`: the seeds together with every intermediate and final structure
    """
    current = list(structures)
    collected = list(current)
    for stage in stages:
        current = list(stage(current))
        collected.extend(current)
    return _dedup(collected)


def depth_first(
    structures: Iterable[Atoms],
    stages: Iterable[Stage],
    executor=None,
) -> list[Atoms]:
    """Push each seed through the whole pipeline independently and collect the result.

    Each seed is run through all `stages` on its own, so the subtree of
    structures derived from one seed never depends on any other seed.  These
    per-seed jobs are therefore embarrassingly parallel: pass an `executor`
    to dispatch them concurrently (e.g. across an HPC allocation), otherwise
    they run sequentially.

    The returned data set is identical to :func:`breadth_first` for the same
    inputs; only the order differs (grouped by seed rather than by stage).

    Args:
        structures (:class:`collections.abc.Iterable` of :class:`ase.Atoms`): the seed structures, e.g. from :func:`~assyst.crystals.sample`
        stages (:class:`collections.abc.Iterable` of :class:`Stage`): the workflow steps to run, in order
        executor (:class:`concurrent.futures.Executor`, optional): anything exposing a compatible ``map``; used to dispatch the per-seed jobs. Defaults to serial execution. ``stages`` must be picklable when a process- or HPC-based executor is used.

    Returns:
        :class:`list` of :class:`ase.Atoms`: the seeds together with every intermediate and final structure
    """
    seeds = list(structures)
    # Materialise so the same stages can drive every independent seed and so
    # they can be shipped to remote workers.
    stages = list(stages)
    run_seed = partial(breadth_first, stages=stages)
    inputs = ([seed] for seed in seeds)
    _map = map if executor is None else executor.map
    collected = []
    for subtree in _map(run_seed, inputs):
        collected.extend(subtree)
    return _dedup(collected)


__all__ = [
    "Stage",
    "RelaxStage",
    "PerturbStage",
    "FilterStage",
    "breadth_first",
    "depth_first",
]
