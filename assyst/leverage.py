"""Training set reduction via statistical leverage scores.

Implements the leverage-guided subset selection of

    A. Khosravi, M. Poul, J. Neugebauer, C. W. Sinclair,
    *Data-Efficient Training of Linear ACE Potentials through Leverage-Guided
    Subset Selection of ASSYST Structure Pools*, `arXiv:2607.18524
    <https://arxiv.org/abs/2607.18524>`__.

The idea is to keep only those structures of a (large) training set that carry the most information for a *linear*
model :math:`E = \\sum_i c \\cdot B_i`, where :math:`B_i` are per-atom features.  Each structure contributes one energy
row (and optionally its force rows) to a design matrix :math:`A`; the diagonal of the regularized hat matrix

.. math:: H_\\lambda = A (A^T A + \\lambda \\Gamma^T \\Gamma)^{-1} A^T

gives per-row leverage scores :math:`h_i \\in [0, 1]` that measure how strongly each observation constrains the fit.
Scores are summed per structure and structures are drawn without replacement with probability proportional to their
score until the requested training fraction is reached.  Selection uses only the feature geometry, never reference
energies or forces, so it can run *before* expensive labeling.

The design matrix requires per-atom features; any :class:`.Featurizer` works.  :class:`.RadialFeaturizer` is a simple
dependency-free default.  Selecting on energy rows only corresponds to the *energy-CUR* mode of the paper, passing a
``force_weight`` adds force rows and corresponds to the *block-CUR* mode.

To see which part of an ASSYST pool a reduction keeps and which it drops, label the structures of each step with
:func:`.tag_stage` and pass the selection to :func:`.trace` or :func:`.summarize`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from typing import Callable, Iterable, Iterator, Sequence, Union

import numpy as np
import pandas as pd
from ase import Atoms

from .neighbors import neighbor_list

STAGE_KEY = "stage"
"""Key in :attr:`ase.Atoms.info` under which :func:`.tag_stage` records the generating ASSYST step."""


class Featurizer(ABC):
    """Compute per-atom features of a structure.

    Features must be linear in the sense that the total energy of a structure is modeled as
    :math:`E = \\sum_i c \\cdot B_i` with one coefficient vector :math:`c` shared by all structures.
    """

    @abstractmethod
    def __call__(self, structure: Atoms) -> np.ndarray:
        """Compute per-atom features.

        Args:
            structure (:class:`ase.Atoms`): structure to featurize

        Returns:
            :class:`numpy.ndarray`: ``(len(structure), n_features)`` array of per-atom features
        """
        pass

    def gradient(self, structure: Atoms) -> np.ndarray:
        """Compute position gradients of the summed features.

        Element ``[k, a, f]`` is :math:`\\partial (\\sum_i B_{if}) / \\partial r_{ka}`, i.e. the negative force design
        row of atom `k` along direction `a`.  Optional; required only to build design matrices with force rows.

        Args:
            structure (:class:`ase.Atoms`): structure to featurize

        Returns:
            :class:`numpy.ndarray`: ``(len(structure), 3, n_features)`` array of gradients
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement feature gradients.")


@dataclass(frozen=True, eq=True)
class RadialFeaturizer(Featurizer):
    """Species-resolved two-body radial basis features.

    For a central atom of species `a` the feature vector holds a constant channel (one per species, capturing the
    atomic reference energy) and, per neighbor species `b`, the radial sums
    :math:`\\phi^{abn}_i = \\sum_{j \\in a_i, |r_{ij}| < r_c} g_n(|r_{ij}|)` over Bessel basis functions
    :math:`g_n(r) = \\sin(n \\pi r / r_c) / r \\cdot f_c(r)` with the smooth cutoff
    :math:`f_c(r) = (1 + \\cos(\\pi r / r_c)) / 2`.

    This is equivalent to a two-body linear ACE and meant as a cheap, dependency-free stand-in for the full ACE basis
    used in the paper.
    """

    elements: tuple[str, ...]
    """Elements the featurizer accepts; determines the feature layout and must cover all featurized structures."""
    n_radial: int = 8
    """Number of radial basis functions per species pair."""
    cutoff: float = 5.0
    """Cutoff radius in Å."""

    def __post_init__(self):
        if len(self.elements) == 0:
            raise ValueError("elements must not be empty!")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError(f"elements must be unique, not {self.elements}!")
        if self.n_radial < 1:
            raise ValueError(f"n_radial must be at least 1, not {self.n_radial}!")
        if self.cutoff <= 0:
            raise ValueError(f"cutoff must be positive, not {self.cutoff}!")

    @property
    def n_features(self) -> int:
        """Total number of features per atom."""
        s = len(self.elements)
        return s * (1 + s * self.n_radial)

    def _species_indices(self, structure: Atoms) -> np.ndarray:
        index = {e: i for i, e in enumerate(self.elements)}
        try:
            return np.array([index[s] for s in structure.symbols])
        except KeyError as e:
            raise ValueError(f"Structure contains element {e} not covered by featurizer {self}!") from None

    def _basis(self, distances: np.ndarray) -> np.ndarray:
        r = distances[:, None]
        k = np.arange(1, self.n_radial + 1)[None, :] * np.pi / self.cutoff
        smooth_cutoff = (1 + np.cos(np.pi * r / self.cutoff)) / 2
        return np.sin(k * r) / r * smooth_cutoff

    def _basis_derivative(self, distances: np.ndarray) -> np.ndarray:
        r = distances[:, None]
        k = np.arange(1, self.n_radial + 1)[None, :] * np.pi / self.cutoff
        smooth_cutoff = (1 + np.cos(np.pi * r / self.cutoff)) / 2
        d_smooth_cutoff = -np.pi / (2 * self.cutoff) * np.sin(np.pi * r / self.cutoff)
        radial = np.sin(k * r) / r
        d_radial = k * np.cos(k * r) / r - np.sin(k * r) / r**2
        return d_radial * smooth_cutoff + radial * d_smooth_cutoff

    def _pair_columns(self, species: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """First feature column of the radial block of each pair, resolved by central and neighbor species."""
        block = 1 + len(self.elements) * self.n_radial
        return species[i] * block + 1 + species[j] * self.n_radial

    def __call__(self, structure: Atoms) -> np.ndarray:
        species = self._species_indices(structure)
        features = np.zeros((len(structure), self.n_features))
        block = 1 + len(self.elements) * self.n_radial
        features[np.arange(len(structure)), species * block] = 1.0
        i, j, d = neighbor_list("ijd", structure, self.cutoff)
        if len(i) > 0:
            columns = self._pair_columns(species, i, j)[:, None] + np.arange(self.n_radial)[None, :]
            np.add.at(features, (i[:, None], columns), self._basis(d))
        return features

    def gradient(self, structure: Atoms) -> np.ndarray:
        species = self._species_indices(structure)
        grad = np.zeros((len(structure), 3, self.n_features))
        i, j, d, D = neighbor_list("ijdD", structure, self.cutoff)
        if len(i) > 0:
            # D points from atom i to atom j, so a pair contributes +g'(d) D/d to the gradient wrt j and -g'(d) D/d
            # to the gradient wrt i
            values = D[:, :, None] / d[:, None, None] * self._basis_derivative(d)[:, None, :]
            columns = (self._pair_columns(species, i, j)[:, None] + np.arange(self.n_radial)[None, :])[:, None, :]
            directions = np.arange(3)[None, :, None]
            np.add.at(grad, (j[:, None, None], directions, columns), values)
            np.add.at(grad, (i[:, None, None], directions, columns), -values)
        return grad


def design_matrix(
    structures: Sequence[Atoms],
    featurizer: Featurizer,
    energy_weight: float = 1.0,
    force_weight: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-structure observation rows of a linear model into a design matrix.

    Each structure contributes one energy row :math:`w_E \\sum_i B_i` and, if `force_weight` is given, ``3 len(s)``
    force rows :math:`-w_F \\partial (\\sum_i B_i) / \\partial r_{ka}`.

    Args:
        structures (:class:`collections.abc.Sequence` of :class:`ase.Atoms`): structures to featurize
        featurizer (:class:`.Featurizer`): computes the per-atom features
        energy_weight (:class:`float`): weight of energy rows
        force_weight (:class:`float` or None): weight of force rows; if None no force rows are included and the
            featurizer needs not implement :meth:`.Featurizer.gradient`

    Returns:
        ``(matrix, owners)``: the design matrix and an integer array mapping each of its rows to the index of the
        structure it belongs to
    """
    if len(structures) == 0:
        raise ValueError("structures must not be empty!")
    rows = []
    owners = []
    for index, structure in enumerate(structures):
        rows.append(energy_weight * featurizer(structure).sum(axis=0))
        owners.append(np.full(1, index))
        if force_weight is not None:
            grad = featurizer.gradient(structure)
            rows.append(-force_weight * grad.reshape(3 * len(structure), -1))
            owners.append(np.full(3 * len(structure), index))
    return np.vstack(rows), np.concatenate(owners)


def leverage_scores(
    matrix: np.ndarray,
    regularization: float = 1e-8,
    tikhonov: np.ndarray | None = None,
) -> np.ndarray:
    """Diagonal of the regularized hat matrix :math:`A (A^T A + \\lambda \\Gamma^T \\Gamma)^{-1} A^T`.

    Args:
        matrix (:class:`numpy.ndarray`): ``(n_rows, n_features)`` design matrix `A`
        regularization (:class:`float`): Tikhonov parameter :math:`\\lambda`; with 0 the classical (pseudo-inverse)
            leverage scores with singular values below numerical rank tolerance discarded are returned.  Scores are
            insensitive to the exact value over many orders of magnitude.
        tikhonov (:class:`numpy.ndarray` or None): ``(n_features,)`` diagonal of the regularization operator
            :math:`\\Gamma`; identity if not given

    Returns:
        :class:`numpy.ndarray`: ``(n_rows,)`` leverage scores, each in ``[0, 1]``
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, not of shape {matrix.shape}!")
    if regularization < 0:
        raise ValueError(f"regularization must be non-negative, not {regularization}!")
    if tikhonov is not None:
        tikhonov = np.asarray(tikhonov, dtype=float)
        if tikhonov.shape != (matrix.shape[1],) or (tikhonov <= 0).any():
            raise ValueError("tikhonov must be a vector of positive values, one per feature!")
        matrix = matrix / tikhonov
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    if regularization == 0:
        keep = s > s.max(initial=0.0) * max(matrix.shape) * np.finfo(float).eps
        return np.einsum("ik,ik->i", u[:, keep], u[:, keep])
    return u**2 @ (s**2 / (s**2 + regularization))


def configuration_leverage(
    structures: Sequence[Atoms],
    featurizer: Featurizer,
    energy_weight: float = 1.0,
    force_weight: float | None = None,
    regularization: float = 1e-8,
    tikhonov: np.ndarray | None = None,
) -> np.ndarray:
    """Total leverage score of each structure.

    Sums the row-level leverage scores of all observations (energy and, if `force_weight` is given, forces) belonging
    to each structure.  Scores depend only on the feature geometry of the pool, not on reference labels.

    Args:
        structures (:class:`collections.abc.Sequence` of :class:`ase.Atoms`): candidate pool
        featurizer (:class:`.Featurizer`): computes the per-atom features
        energy_weight (:class:`float`): weight of energy rows
        force_weight (:class:`float` or None): weight of force rows; if None only energy rows enter
        regularization (:class:`float`): Tikhonov parameter, see :func:`.leverage_scores`
        tikhonov (:class:`numpy.ndarray` or None): diagonal regularization operator, see :func:`.leverage_scores`

    Returns:
        :class:`numpy.ndarray`: ``(len(structures),)`` non-negative structure scores
    """
    matrix, owners = design_matrix(structures, featurizer, energy_weight=energy_weight, force_weight=force_weight)
    scores = leverage_scores(matrix, regularization=regularization, tikhonov=tikhonov)
    return np.bincount(owners, weights=scores, minlength=len(structures))


def sample_without_replacement(
    weights: np.ndarray,
    number: int,
    rng: Union[int, np.random.Generator, None] = None,
) -> np.ndarray:
    """Draw indices without replacement with probability proportional to `weights`.

    Successively draws one index at a time with probabilities renormalized over the remaining ones.

    Args:
        weights (:class:`numpy.ndarray`): non-negative selection weights
        number (:class:`int`): how many indices to draw
        rng (:class:`int`, :class:`numpy.random.Generator`): seed or random number generator

    Returns:
        :class:`numpy.ndarray`: ``(number,)`` drawn indices, in draw order

    Raises:
        ValueError: if weights are negative or fewer weights are positive than draws are requested
    """
    weights = np.asarray(weights, dtype=float).copy()
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, not of shape {weights.shape}!")
    if (weights < 0).any():
        raise ValueError("weights must be non-negative!")
    if not 0 <= number <= len(weights):
        raise ValueError(f"number must be in range [0, {len(weights)}], not {number}!")
    positive = np.count_nonzero(weights)
    if number > positive:
        raise ValueError(f"Cannot draw {number} indices, only {positive} weights are positive!")
    rng = np.random.default_rng(rng)
    drawn = np.empty(number, dtype=int)
    for k in range(number):
        drawn[k] = rng.choice(len(weights), p=weights / weights.sum())
        weights[drawn[k]] = 0.0
    return drawn


def select(
    structures: Iterable[Atoms],
    fraction: float | None = None,
    number: int | None = None,
    featurizer: Featurizer | None = None,
    energy_weight: float = 1.0,
    force_weight: float | None = None,
    regularization: float = 1e-8,
    tikhonov: np.ndarray | None = None,
    rng: Union[int, np.random.Generator, None] = None,
) -> np.ndarray:
    """Select a leverage-guided subset of a structure pool.

    Computes per-structure leverage scores and samples structures without replacement with probability proportional
    to their score.  With the default ``force_weight=None`` this is the *energy-CUR* mode of the paper; passing e.g.
    ``force_weight=0.1`` (together with the default ``energy_weight=1.0``) adds force rows to the design matrix and
    corresponds to the *block-CUR* mode.

    >>> pool = [...]                               # doctest: +SKIP
    >>> subset = [pool[i] for i in select(pool, fraction=0.3, rng=42)]    # doctest: +SKIP

    Args:
        structures (:class:`collections.abc.Iterable` of :class:`ase.Atoms`): candidate pool
        fraction (:class:`float`): fraction of the pool to select, in ``(0, 1]``; rounded up to full structures
        number (:class:`int`): number of structures to select; give either this or `fraction`
        featurizer (:class:`.Featurizer` or None): computes the per-atom features; by default a
            :class:`.RadialFeaturizer` covering all elements in the pool
        energy_weight (:class:`float`): weight of energy rows
        force_weight (:class:`float` or None): weight of force rows; if None only energy rows enter and the
            featurizer needs not implement gradients
        regularization (:class:`float`): Tikhonov parameter, see :func:`.leverage_scores`
        tikhonov (:class:`numpy.ndarray` or None): diagonal regularization operator, see :func:`.leverage_scores`
        rng (:class:`int`, :class:`numpy.random.Generator`): seed or random number generator

    Returns:
        :class:`numpy.ndarray`: indices of the selected structures into the pool, in draw order

    Raises:
        ValueError: if not exactly one of `fraction` and `number` is given, or it is out of range
    """
    structures = list(structures)
    if (fraction is None) == (number is None):
        raise ValueError("Give exactly one of fraction and number!")
    if fraction is not None:
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in range (0, 1], not {fraction}!")
        number = ceil(fraction * len(structures))
    if featurizer is None:
        featurizer = RadialFeaturizer(tuple(sorted({str(sym) for s in structures for sym in s.symbols})))
    scores = configuration_leverage(
        structures,
        featurizer,
        energy_weight=energy_weight,
        force_weight=force_weight,
        regularization=regularization,
        tikhonov=tikhonov,
    )
    return sample_without_replacement(scores, number, rng=rng)


def tag_stage(structures: Iterable[Atoms], stage: str) -> Iterator[Atoms]:
    """Record which ASSYST step produced each structure.

    Writes `stage` to ``structure.info[STAGE_KEY]`` and yields the structures again, so it can be wrapped around any
    step of a pipeline.  ASSYST itself does not track this: perturbations leave a ``"perturbation"`` entry in
    :attr:`~ase.Atoms.info` (which :func:`.stage_of` falls back to), but the relaxation steps are not distinguishable
    after the fact.

    Operates INPLACE.

    >>> volmin = list(tag_stage(relax(seeds, VolumeRelax(), calc), "volmin"))    # doctest: +SKIP

    Args:
        structures (:class:`collections.abc.Iterable` of :class:`ase.Atoms`): structures to tag
        stage (:class:`str`): name of the step that produced them

    Yields:
        :class:`ase.Atoms`: the same structures, tagged
    """
    for structure in structures:
        structure.info[STAGE_KEY] = stage
        yield structure


def stage_of(structure: Atoms) -> str:
    """Name of the ASSYST step a structure came from.

    Returns the tag written by :func:`.tag_stage` if present, otherwise the perturbation recorded by
    :class:`~assyst.perturbations.PerturbationABC`, otherwise ``"unknown"``.

    Args:
        structure (:class:`ase.Atoms`): structure to inspect

    Returns:
        :class:`str`: the step name
    """
    if STAGE_KEY in structure.info:
        return str(structure.info[STAGE_KEY])
    if "perturbation" in structure.info:
        return str(structure.info["perturbation"])
    return "unknown"


def trace(
    structures: Sequence[Atoms],
    selected: Iterable[int],
    scores: np.ndarray | None = None,
    stage: Callable[[Atoms], str] = stage_of,
    **kwargs,
) -> pd.DataFrame:
    """Tabulate which structures a selection keeps and which it discards.

    Args:
        structures (:class:`collections.abc.Sequence` of :class:`ase.Atoms`): the pool that was selected from
        selected (:class:`collections.abc.Iterable` of :class:`int`): indices returned by :func:`.select`
        scores (:class:`numpy.ndarray` or None): per-structure leverage scores; computed with
            :func:`.configuration_leverage` if not given
        stage (callable): maps a structure to the name of the step that produced it; :func:`.stage_of` by default
        **kwargs: passed to :func:`.configuration_leverage` when `scores` is not given

    Returns:
        :class:`pandas.DataFrame`: one row per structure, with columns ``stage``, ``selected``, ``rank`` (position in
        the draw order, ``-1`` if discarded), ``score``, ``number_of_atoms``, ``volume_per_atom`` and ``formula``
    """
    if scores is None:
        scores = configuration_leverage(structures, **kwargs)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (len(structures),):
        raise ValueError(f"scores must hold one value per structure, not {scores.shape}!")
    rank = np.full(len(structures), -1)
    for position, index in enumerate(selected):
        rank[index] = position
    return pd.DataFrame(
        {
            "stage": [stage(s) for s in structures],
            "selected": rank >= 0,
            "rank": rank,
            "score": scores,
            "number_of_atoms": [len(s) for s in structures],
            "volume_per_atom": [s.cell.volume / len(s) for s in structures],
            "formula": [s.get_chemical_formula() for s in structures],
        }
    )


def summarize(trace_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a :func:`.trace` per ASSYST step.

    Args:
        trace_frame (:class:`pandas.DataFrame`): as returned by :func:`.trace`

    Returns:
        :class:`pandas.DataFrame`: one row per step, indexed by stage, with the number of structures in the pool, how
        many were ``selected`` and ``discarded``, the ``selected_fraction`` of that step, the step's share of the
        total leverage of the pool (``score_share``) and its ``mean_score``
    """
    grouped = trace_frame.groupby("stage")
    total = trace_frame["score"].sum()
    summary = pd.DataFrame(
        {
            "pool": grouped.size(),
            "selected": grouped["selected"].sum(),
            "mean_score": grouped["score"].mean(),
            "score_share": grouped["score"].sum() / total if total > 0 else 0.0,
        }
    )
    summary["discarded"] = summary["pool"] - summary["selected"]
    summary["selected_fraction"] = summary["selected"] / summary["pool"]
    return summary[["pool", "selected", "discarded", "selected_fraction", "mean_score", "score_share"]]


__all__ = [
    "Featurizer",
    "RadialFeaturizer",
    "STAGE_KEY",
    "design_matrix",
    "leverage_scores",
    "configuration_leverage",
    "sample_without_replacement",
    "select",
    "tag_stage",
    "stage_of",
    "trace",
    "summarize",
]
