'''Classes that filter structures according to some criteria.

The code in the other modules that uses them is set up such that simple
functions can always be passed as well and that the classes here are just for
convenience.'''

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, KW_ONLY
from itertools import combinations_with_replacement, product
from math import nan, inf
from typing import Callable, Literal

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from structuretoolkit import get_neighbors
from pyxtal.tolerance import Tol_matrix
from ase.data import atomic_numbers
import numpy as np


class FilterBase(ABC):
    '''Base class for filter objects that implements conjunction and disjunction operators.'''
    def __and__(self, other) -> 'AndFilter':
        return AndFilter(self, other)

    def __or__(self, other) -> 'OrFilter':
        return OrFilter(self, other)

    @abstractmethod
    def __call__(self, structure: Atoms) -> bool:
        '''Returns True if structure passes the filter, False if it should be dropped.'''
        pass


Filter = Callable[[Atoms], bool] | FilterBase


@dataclass(frozen=True, eq=True)
class AndFilter(FilterBase):
    '''Conjunction of two filters.'''
    l: Filter
    r: Filter

    def __call__(self, structure: Atoms) -> bool:
        return self.l(structure) and self.r(structure)


@dataclass(frozen=True, eq=True)
class OrFilter(FilterBase):
    '''Disjunction of two filters.'''
    l: Filter
    r: Filter

    def __call__(self, structure: Atoms) -> bool:
        return self.l(structure) or self.r(structure)


@dataclass
class DistanceFilter(FilterBase):
    '''Filter structures that contain too close atoms.

    Setting a radius to NaN allows all bonds involving this atom.'''
    radii: dict[str, float]

    @staticmethod
    def _element_wise_dist(structure: Atoms) -> dict[tuple[str, str], float]:
        pair: dict[tuple[str, str], float] = defaultdict(lambda: inf)
        # on weird aspect ratios the neighbor searching code can allocate huge structures,
        # because it explicitly repeats the structure to create ghost atoms
        # since we only care about the presence of short distances between atoms and not the
        # real neighbor information, simply double the structure to make sure we see all bonds
        # and turn off PBC
        # this can miss neighbors in structures with highly acute lattice angles, but we'll live
        sr = structure.repeat(2)
        sr.pbc = [False, False, False]
        n = get_neighbors(sr, num_neighbors=len(structure), mode="ragged")
        for i, (I, D) in enumerate(zip(n.indices, n.distances)):
            for j, d in zip(I, D):
                ei, ej = sorted((sr.symbols[i], sr.symbols[j]))
                pair[ei, ej] = min(d, pair[ei, ej])
        return pair

    def __call__(self, structure: Atoms) -> bool:
        '''
        Return True if structure satifies minimum distance criteria.

        Args:
            structure (ase.Atoms): structure to check

        Returns:
            `False`: at least on bond is shorter than the sum of given cutoff radii of the respective elements
            `True`: all bonds are than the sum of given cutoff radii of the respective elements
        '''
        pair = self._element_wise_dist(structure)
        for ei, ej in combinations_with_replacement(structure.symbols.species(), 2):
            ei, ej = sorted((ei, ej))
            if pair[ei, ej] < self.radii.get(ei, nan) + self.radii.get(ej, nan):
                return False
        return True

    def to_tol_matrix(self, prototype: Literal['metallic', 'atomic', 'molecular', 'vdW'] = 'metallic') -> Tol_matrix:
        '''Returns equivalent tolerance matrix for pyxtal.

        Args:
            prototype (metallic, atomic, molecular or vdW):
                passed to Tol_matrix as is and used there to initialize radii of elements not explicitly set in this
                filter
        '''
        return Tol_matrix(
                *((atomic_numbers[e1], atomic_numbers[e2], self.radii[e1] + self.radii[e2])
                    for e1, e2 in product(self.radii, repeat=2)),
                prototype=prototype
        )


@dataclass
class AspectFilter(FilterBase):
    '''Filters structures with high aspect ratios.'''
    maximum_aspect_ratio: float = 6

    def __call__(self, structure: Atoms) -> bool:
        '''Return True if structure's cell has an agreeable aspect ratio.

        Args:
            structure (ase.Atoms): structure to check

        Returns:
            `True`: lattice's aspect ratio is below or equal `:attr:`.maximum_aspect_ratio`.
            `False`: lattice's aspect ratio is above `:attr:`.maximum_aspect_ratio`.'''
        a, b, c = sorted(structure.cell.lengths())
        return c / a <= self.maximum_aspect_ratio


@dataclass
class VolumeFilter(FilterBase):
    '''Filters structures by volume.'''
    maximum_volume_per_atom: float

    def __call__(self, structure: Atoms) -> bool:
        '''Return True if structure's volume is within range.

        Args:
            structure (ase.Atoms): structure to check

        Returns:
            `True`: volume per atom is smaller or equal than `:attr:.maximum_volume_per_atom`.
            `False`: otherwise'''
        return structure.cell.volume / len(structure) <= self.maximum_volume_per_atom


@dataclass
class CalculatorFilter(FilterBase):
    '''Filters that require a single point calculator set on the structure.'''

    _: KW_ONLY
    missing: Literal['error', 'ignore'] = 'error'
    '''What to do when a structure has no (correct) calculator attached.'''

    def _check(self, structure: Atoms) -> bool:
        match self.missing:
            case 'error':
                if structure.calc is None:
                    raise ValueError('Structure must have single point calculator set!')
                if not isinstance(structure.calc, SinglePointCalculator):
                    raise ValueError(f'Structure must have single point calculator set, not {type(structure.calc)}!')
                return True
            case 'ignore':
                return False
            case _:
                assert False


@dataclass
class EnergyFilter(CalculatorFilter):
    '''Filters structures by energy per atom.'''
    min_energy: float = -inf
    max_energy: float = +inf

    def __call__(self, structure: Atoms) -> bool:
        if not self._check(structure):
            return True
        return self.min_energy <= structure.get_potential_energy() / len(structure) <= self.max_energy


@dataclass
class ForceFilter(CalculatorFilter):
    '''Filters structures by maximum force magnitude.'''
    max_force: float = +inf

    def __call__(self, structure: Atoms) -> bool:
        if not self._check(structure):
            return True
        return np.linalg.norm(structure.get_forces(), axis=-1).max() <= self.max_force
