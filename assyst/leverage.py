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
dependency-free default, :class:`.AceFeaturizer` wraps the linear ACE basis the paper itself uses and needs `pyace`
installed.  Selecting on energy rows only corresponds to the *energy-CUR* mode of the paper, passing a
``force_weight`` adds force rows and corresponds to the *block-CUR* mode.

To see which part of an ASSYST pool a reduction keeps and which it drops, pass the selection to :func:`.trace` or
:func:`.summarize`.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Callable, Iterable, Sequence, Union

import numpy as np
import pandas as pd
from ase import Atoms
from pyiron_snippets.import_alarm import ImportAlarm

from .neighbors import neighbor_list
from .utils import stage_of as _workflow_stage_of

with ImportAlarm(
    "AceFeaturizer requires pyace; install with 'conda install -c conda-forge python-ace' or from "
    "https://github.com/ICAMS/python-ace",
    raise_exception=True,
) as ace_alarm:
    import pyace
    from pyace.atomicenvironment import aseatoms_to_atomicenvironment  # needs pyace
    from pyace.linearacefit import compute_nfunc_func_ind_shift  # needs pyace


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


PACEMAKER_FUNCTIONS = {
    "UNARY": {"nradmax_by_orders": (15, 6, 4, 3, 2, 2), "lmax_by_orders": (0, 3, 3, 2, 2, 1)},
    "BINARY": {"nradmax_by_orders": (15, 6, 3, 2, 2, 1), "lmax_by_orders": (0, 3, 2, 1, 1, 0)},
    "TERNARY": {"nradmax_by_orders": (15, 3, 3, 2, 1), "lmax_by_orders": (0, 2, 2, 1, 1)},
    "ALL": {"nradmax_by_orders": (15, 3, 2, 1, 1), "lmax_by_orders": (0, 2, 2, 1, 1)},
}
"""Basis shape per bond block that ``pacemaker -t`` writes into its ``input.yaml`` template."""


@ace_alarm
def _load_basis(potential: str):  # needs pyace
    """Read a B-basis from a potential file, with a hint when it is a C-tilde file instead."""
    try:
        return pyace.ACEBBasisSet(potential)
    except Exception as e:
        raise ValueError(
            f"Could not read a B-basis from {potential}; pacemaker writes one as 'output_potential.yaml', "
            "while '.yace' files hold the C-tilde basis, which cannot give B-basis projections."
        ) from e


def _functions_block(featurizer: "AceFeaturizer") -> dict:  # needs pyace
    """The ``functions`` section of the pyace configuration."""
    if featurizer.n_radial is None:
        functions = {block: dict(spec) for block, spec in PACEMAKER_FUNCTIONS.items()}
    else:
        block = "UNARY" if len(featurizer.elements) == 1 else "ALL"
        functions = {block: {"nradmax_by_orders": featurizer.n_radial, "lmax_by_orders": featurizer.l_max}}
    functions = {b: {k: list(v) for k, v in spec.items()} for b, spec in functions.items()}
    if featurizer.n_functions_per_element is not None:
        functions["number_of_functions_per_element"] = featurizer.n_functions_per_element
    return functions


@lru_cache(maxsize=4)
def _ace_engine(featurizer: "AceFeaturizer"):  # needs pyace
    """Build the (expensive, unpicklable) pyace objects for a featurizer and keep them cached.

    Held outside :class:`.AceFeaturizer` so that it stays a plain, picklable dataclass.
    """
    if featurizer.potential is not None:
        basis = pyace.ACEBBasisSet(featurizer.potential)
        stored = tuple(basis.elements_name)
        if stored != tuple(featurizer.elements):
            raise ValueError(
                f"Potential {featurizer.potential} is fitted for elements {stored}, "
                f"but featurizer declares {tuple(featurizer.elements)}!"
            )
    else:
        configuration = pyace.create_multispecies_basis_config(
            {
                "elements": list(featurizer.elements),
                "deltaSplineBins": 0.001,
                # the embedding is the non-linearity and does not enter the B-basis projections at all; a linear one
                # is used so the featurizer matches the model that consumes it
                "embeddings": {"ALL": {"fs_parameters": [1, 1], "ndensity": 1, "npot": "FinnisSinclair"}},
                "bonds": {
                    "ALL": {
                        "NameOfCutoffFunction": "cos",
                        "dcut": 0.01,
                        "radbase": featurizer.radial_base,
                        "radparameters": list(featurizer.radial_parameters),
                        "rcut": featurizer.cutoff,
                    }
                },
                "functions": _functions_block(featurizer),
            }
        )
        basis = pyace.ACEBBasisSet(configuration)
    # the calculator keeps a raw pointer to the evaluator and the evaluator one to the basis, so all three have to
    # be kept alive together
    evaluator = pyace.ACEBEvaluator(basis)
    calculator = pyace.ACECalculator(evaluator)
    n_features, shifts = compute_nfunc_func_ind_shift(basis)
    return basis, evaluator, calculator, n_features, shifts


@dataclass(frozen=True, eq=True)
class AceFeaturizer(Featurizer):  # needs pyace
    """Linear Atomic Cluster Expansion features, as used in the paper.

    Wraps the B-basis of `pyace <https://github.com/ICAMS/python-ace>`__: :meth:`~.AceFeaturizer.__call__` returns
    the per-atom B-basis projections and :meth:`~.AceFeaturizer.gradient` their position derivatives, so both
    energy-CUR and block-CUR selection work.  Species are laid out in blocks, one per element.

    By default the basis is the one ``pacemaker -t`` writes into its ``input.yaml`` template
    (:data:`.PACEMAKER_FUNCTIONS`), but *without* its ``number_of_functions_per_element`` filter, so the full basis
    is used.  Set :attr:`.n_functions_per_element` to truncate it the way a template-generated fit would; that is
    worth doing for large pools, since the untruncated basis has 945 functions for a unary and 4452 for a binary
    system.

    An existing potential can be used as the basis specification instead, which is the way to score structures on
    exactly the basis a fit already uses::

        featurizer = AceFeaturizer.from_potential("output_potential.yaml")

    .. attention::
        This class needs additional dependencies!
        Install `pyace` with ``conda install -c conda-forge python-ace`` or from
        `Github <https://github.com/ICAMS/python-ace>`__.
    """

    elements: tuple[str, ...]
    """Elements the featurizer accepts; determines the feature layout and must cover all featurized structures."""
    n_radial: tuple[int, ...] | None = None
    """Number of radial functions per body order, i.e. ``nradmax_by_orders``; its length sets the body order.

    ``None`` uses the per-block :data:`.PACEMAKER_FUNCTIONS` defaults, otherwise the given shape applies to all bonds.
    """
    l_max: tuple[int, ...] | None = None
    """Maximum angular momentum per body order, i.e. ``lmax_by_orders``; must be as long as :attr:`.n_radial`."""
    cutoff: float = 7.0
    """Cutoff radius in Å; the ``pacemaker -t`` default."""
    radial_base: str = "SBessel"
    """Radial basis family passed to pyace."""
    radial_parameters: tuple[float, ...] = (5.25,)
    """Parameters of the radial basis family, i.e. ``radparameters``."""
    n_functions_per_element: int | None = None
    """Keep only this many functions per element, i.e. ``number_of_functions_per_element``; ``None`` keeps all."""
    potential: str | None = None
    """Path to a B-basis potential file to take the basis from; overrides all other basis settings.

    Prefer :meth:`.from_potential`, which fills :attr:`.elements` from the file.
    """

    @classmethod
    def from_potential(cls, potential: Union[str, "os.PathLike"], **kwargs) -> "AceFeaturizer":
        """Take the basis from an existing ACE potential, so structures are scored on the basis a fit already uses.

        Args:
            potential (str): path to a B-basis potential file, i.e. pacemaker's ``output_potential.yaml``; the
                ``.yace`` C-tilde files written for LAMMPS cannot provide B-basis projections
            **kwargs: passed to the constructor; basis settings are ignored in favour of the file

        Returns:
            :class:`.AceFeaturizer`: featurizer over the elements the potential was fitted for
        """
        potential = str(potential)
        if not os.path.exists(potential):
            raise FileNotFoundError(f"No such potential file: {potential}!")
        return cls(elements=tuple(_load_basis(potential).elements_name), potential=potential, **kwargs)

    @ace_alarm
    def __post_init__(self):
        if len(self.elements) == 0:
            raise ValueError("elements must not be empty!")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError(f"elements must be unique, not {self.elements}!")
        if self.n_functions_per_element is not None and self.n_functions_per_element < 1:
            raise ValueError(
                f"n_functions_per_element must be positive, not {self.n_functions_per_element}!"
            )
        if self.cutoff <= 0:
            raise ValueError(f"cutoff must be positive, not {self.cutoff}!")
        if self.potential is not None:
            if not os.path.exists(self.potential):
                raise FileNotFoundError(f"No such potential file: {self.potential}!")
            return
        if (self.n_radial is None) != (self.l_max is None):
            raise ValueError("n_radial and l_max must be given together, or neither of them!")
        if self.n_radial is None:
            return
        if len(self.n_radial) != len(self.l_max):
            raise ValueError(
                f"n_radial and l_max must be of same length, not {self.n_radial} and {self.l_max}!"
            )
        if len(self.n_radial) == 0:
            raise ValueError("n_radial must not be empty!")
        if any(n < 1 for n in self.n_radial):
            raise ValueError(f"n_radial must be positive, not {self.n_radial}!")
        if any(lmax < 0 for lmax in self.l_max):
            raise ValueError(f"l_max must be non-negative, not {self.l_max}!")
        if self.l_max[0] != 0:
            raise ValueError(f"The rank-1 block is radial only, so l_max[0] must be 0, not {self.l_max[0]}!")

    @property
    def n_features(self) -> int:
        """Total number of basis functions over all species."""
        return _ace_engine(self)[3]

    def _evaluate(self, structure: Atoms, gradient: bool):
        basis, _, calculator, _, shifts = _ace_engine(self)
        unknown = set(structure.symbols) - set(self.elements)
        if unknown:
            raise ValueError(f"Structure contains elements {sorted(unknown)} not covered by featurizer {self}!")
        environment = aseatoms_to_atomicenvironment(
            structure, cutoff=basis.cutoffmax, elements_mapper_dict=basis.elements_to_index_map
        )
        calculator.compute(environment, compute_projections=True, compute_b_grad=gradient)
        return environment, calculator, shifts

    def __call__(self, structure: Atoms) -> np.ndarray:
        environment, calculator, shifts = self._evaluate(structure, gradient=False)
        features = np.zeros((len(structure), self.n_features))
        for atom, (species, projection) in enumerate(zip(environment.species_type, calculator.projections)):
            features[atom, shifts[species]: shifts[species] + len(projection)] = projection
        return features

    def gradient(self, structure: Atoms) -> np.ndarray:
        _, calculator, _ = self._evaluate(structure, gradient=True)
        # pyace returns the force per unit coefficient, i.e. the negative gradient, as (atoms, features, directions)
        return -np.asarray(calculator.forces_bfuncs).transpose((0, 2, 1))


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


def stage_of(structure: Atoms) -> str:
    """The ASSYST steps a structure went through, as recorded in its metadata.

    Reads the ``stage`` key that every step of the workflow appends to, so generated, volume relaxed and fully
    relaxed structures report ``spg``, ``spg+volume_relax`` and ``spg+volume_relax+full_relax`` respectively, and
    perturbed ones carry their perturbation on top.  Structures that ASSYST did not make report ``"unknown"``.

    Reporting the full history keeps the steps apart, but is more detail than a summary usually wants; pass your own
    function to :func:`.trace` to coarsen it, e.g. to the last step only.

    Args:
        structure (:class:`ase.Atoms`): structure to inspect

    Returns:
        :class:`str`: the steps, joined with ``+``

    See Also:
        :func:`assyst.utils.stage_of`: the reader this delegates to
    """
    return _workflow_stage_of(structure)


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
    "AceFeaturizer",
    "design_matrix",
    "leverage_scores",
    "configuration_leverage",
    "sample_without_replacement",
    "select",
    "stage_of",
    "trace",
    "summarize",
]
