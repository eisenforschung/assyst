'''Classes to apply (random) perturbations to structures.'''

from abc import ABC, abstractmethod
from ase import Atoms
from typing import Iterable, Callable, Self, Iterator
from dataclasses import dataclass
import numpy as np

from .filters import Filter


def rattle(structure: Atoms, sigma: float) -> Atoms:
    """Randomly displace positions with gaussian noise.

    Operates INPLACE."""
    if len(structure) == 1:
        raise ValueError("Can only rattle structures larger than one atom.")
    structure.rattle(stdev=sigma)
    return structure


def stretch(structure: Atoms, hydro: float, shear: float, minimum_strain=1e-3) -> Atoms:
    """Randomly stretch cell with uniform noise.

    Ensures at least `minimum_strain` strain to avoid structures very close to their original structures.
    These don't offer a lot of new information and can also confuse VASP's symmetry analyzer.

    Operates INPLACE."""
    def get_strains(max_strain, size):
        signs = np.random.choice([-1, 1], size=size)
        magnitudes = np.random.uniform(minimum_strain, max_strain, size=size)
        return signs * magnitudes

    strain = np.zeros((3,3))
    # Off-diagonal elements
    indices = np.triu_indices(3, k=1)
    strain[indices] = get_strains(shear, 3)
    strain += strain.T

    # Diagonal elements
    np.fill_diagonal(strain, 1 + get_strains(hydro, 3))

    structure.set_cell(structure.cell.array @ strain, scale_atoms=True)
    return structure


class PerturbationABC(ABC):
    """Apply some perturbation to a given structure."""
    def __call__(self, structure: Atoms) -> Atoms:
        if 'perturbation' not in structure.info:
            structure.info['perturbation'] = str(self)
        else:
            structure.info['perturbation'] += '+' + str(self)
        return structure

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __add__(self, other: Self) -> 'Series':
        return Series((self, other))


Perturbation = Callable[[Atoms], Atoms] | PerturbationABC


def apply_perturbations(
        structures: Iterable[Atoms],
        perturbations: Iterable[Perturbation],
        filters: Iterable[Filter] | Filter | None = None,
) -> Iterator[Atoms]:
    '''Apply a list of perturbations to each structure and yield the result of each perturbation separately.

    If a perturbation raises ValueError it is ignored.'''
    if filters is None:
        filters = []
    if not isinstance(filters, Iterable):
        filters = [filters]
    perturbations = list(perturbations)

    for structure in structures:
        for mod in perturbations:
            try:
                m = mod(structure.copy())
            except ValueError:
                continue
            if all(f(m) for f in filters):
                yield m


@dataclass(frozen=True)
class Rattle(PerturbationABC):
    """Displace atoms by some absolute amount from a normal distribution."""
    sigma: float
    create_supercells: bool = False
    "Create minimal 2x2x2 super cells when applied to structures of only one atom."

    def __call__(self, structure: Atoms):
        if self.create_supercells and len(structure) == 1:
            structure = structure.repeat(2)
        structure = super().__call__(structure)
        return rattle(structure, self.sigma)

    def __str__(self):
        return f"rattle({self.sigma})"


@dataclass(frozen=True)
class Stretch(PerturbationABC):
    """Apply random cell perturbation."""
    hydro: float
    shear: float
    minimum_strain: float = 1e-3

    def __call__(self, structure: Atoms):
        structure = super().__call__(structure)
        return stretch(structure, self.hydro, self.shear, self.minimum_strain)

    def __str__(self):
        return f"stretch(hydro={self.hydro}, shear={self.shear})"


@dataclass(frozen=True)
class Series(PerturbationABC):
    """Apply some perturbations in sequence."""
    perturbations: tuple[Perturbation, ...]

    def __call__(self, structure: Atoms) -> Atoms:
        for mod in self.perturbations:
            structure = mod(structure)
        return structure

    def __str__(self):
        return "+".join(str(mod) for mod in self.perturbations)


@dataclass(frozen=True)
class RandomChoice(PerturbationABC):
    """Apply either of two alternatives randomly."""
    choice_a: Perturbation
    choice_b: Perturbation
    chance: float
    "Probability to pick choice b"

    def __call__(self, structure: Atoms) -> Atoms:
        if np.random.rand() > self.chance:
            return self.choice_a(structure)
        else:
            return self.choice_b(structure)

    def __str__(self):
        return str(self.choice_a) + "|" + str(self.choice_b)
