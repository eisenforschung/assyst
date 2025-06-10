"""Nodes to randomly rattle and shake structures."""

from abc import ABC, abstractmethod
from ase import Atoms
from typing import Iterable, Callable, NewType, Self, Iterator
from dataclasses import dataclass
import numpy as np


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
    def scale(r):
        # map [0, 1) to [-1, 1)
        r = 2 * r - 1
        # map [-1, 1) to [-(1-m), 1-m)
        r *= (1 - minimum_strain)
        # map  [-(1-m), 1-m) to [-1, -m] u [m, 1)
        r += np.sign(r) * minimum_strain
        return r

    strain = shear * (2 * np.random.rand(3, 3) - 1)
    strain = 0.5 * (strain + strain.T)  # symmetrize
    np.fill_diagonal(strain, 1 + hydro * (2 * np.random.rand(3) - 1))
    structure.set_cell(structure.cell.array @ strain, scale_atoms=True)
    return structure


class ModifyABC(ABC):
    """Apply some modification to a given structure."""
    def __call__(self, structure: Atoms) -> Atoms:
        if 'modification' not in structure.info:
            structure.info['modification'] = str(self)
        else:
            structure.info['modification'] += '+' + str(self)
        return structure

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __add__(self, other: Self) -> 'Series':
        return Series((self, other))

Filter = Callable[[Atoms], bool]
Modify = Callable[[Atoms], Atoms] | ModifyABC

def apply_modifications(
        structures: Iterable[Atoms],
        modifications: Iterable[Modify],
        filters: Iterable[Filter] | Filter | None = None,
) -> Iterator[Atoms]:
    """Apply a list of modifications to each structure and yield the result of each modification separately."""
    if filters is None:
        filters = []
    if not isinstance(filters, Iterable):
        filters = [filters]
    modifications = list(modifications)

    for structure in structures:
        for mod in modifications:
            try:
                m = mod(structure.copy())
            except ValueError:
                continue
            if all(f(m) for f in filters):
                yield m


@dataclass(frozen=True)
class Rattle(ModifyABC):
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
class Stretch(ModifyABC):
    """Apply random cell modification."""
    hydro: float
    shear: float
    minimum_strain: float = 1e-3

    def __call__(self, structure: Atoms):
        structure = super().__call__(structure)
        return stretch(structure, self.hydro, self.shear, self.minimum_strain)

    def __str__(self):
        return f"stretch(hydro={self.hydro}, shear={self.shear})"


@dataclass(frozen=True)
class Series(ModifyABC):
    """Apply some modifications in sequence."""
    modifications: tuple[Modify, ...]

    def __call__(self, structure: Atoms) -> Atoms:
        for mod in self.modifications:
            structure = mod(structure)
        return structure

    def __str__(self):
        return "+".join(str(mod) for mod in self.modifications)


@dataclass(frozen=True)
class RandomChoice(ModifyABC):
    """Apply either of two alternatives randomly."""
    choice_a: Modify
    choice_b: Modify
    chance: float

    def __call__(self, structure: Atoms) -> Atoms:
        if np.random.rand() > self.chance:
            return self.choice_a(structure)
        else:
            return self.choice_b(structure)

    def __str__(self):
        return str(self.choice_a) + "|" + str(self.choice_b)
