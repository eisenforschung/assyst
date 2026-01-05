"""Classes to apply (random) perturbations to structures."""

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
    structure.rattle(stdev=sigma, rng=np.random)
    return structure


def element_scaled_rattle(structure: Atoms, sigma: float, reference: dict[str, float]) -> Atoms:
    """Randomly displace positions with gaussian noise relative to an elemental reference length.

    Operates like :func:`.rattle` but uses a standard deviation derived from the relative `sigma` and the `reference`,
    where this reference is given by element.

    Operates IN PLACE!

    Args:
        structure (:class:`.ase.Atoms`): structure to perturb
        sigma (float): relative standard deviation
        reference (dict of str to float): reference length per element

    Raises:
        ValueError: if len(structure) == 1, create a super cell first before calling again
        ValueError: if reference values are not positive
        ValueError: if reference does not contain all elements in given structure
    """
    sigma = sigma * np.ones(len(structure))
    if not all(r > 0 for r in reference.values()):
        raise ValueError("Reference lengths must be strictly positive!")
    for i, sym in enumerate(structure.symbols):
        try:
            sigma[i] *= reference[sym]
        except KeyError:
            raise ValueError(f"No value for element {sym} provided in argument `reference`!") from None
    return rattle(structure, sigma.reshape(-1, 1))


def stretch(structure: Atoms, hydro: float, shear: float, minimum_strain=1e-3) -> Atoms:
    """Randomly stretch cell with uniform noise.

    Ensures at least `minimum_strain` strain to avoid structures very close to their original structures.
    These don't offer a lot of new information and can also confuse VASP's symmetry analyzer.

    Operates INPLACE."""

    def get_strains(max_strain, size):
        signs = np.random.choice([-1, 1], size=size)
        magnitudes = np.random.uniform(minimum_strain, max_strain, size=size)
        return signs * magnitudes

    strain = np.zeros((3, 3))
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
        if "perturbation" not in structure.info:
            structure.info["perturbation"] = str(self)
        else:
            structure.info["perturbation"] += "+" + str(self)
        return structure

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __add__(self, other: Self) -> "Series":
        return Series((self, other))


Perturbation = Callable[[Atoms], Atoms] | PerturbationABC


def apply_perturbations(
    structures: Iterable[Atoms],
    perturbations: Iterable[Perturbation],
    filters: Iterable[Filter] | Filter | None = None,
    retries: int = 10,
) -> Iterator[Atoms]:
    """Apply a list of perturbations to each structure and yield the result of each perturbation separately.

    If a perturbation raises ValueError it is ignored.

    Args:
        structures: Atomic structures to perturb.
        perturbations: Functions that modify structures.
        filters: Functions to filter valid results (optional).
        retries: Max attempts per perturbation (default: 10).

    Yields:
        Perturbed structure that passes all filters.
    """
    if filters is None:
        filters = []
    if not isinstance(filters, Iterable):
        filters = [filters]
    perturbations = list(perturbations)

    for structure in structures:
        for mod in perturbations:
            try:
                for _ in range(retries):
                    m = mod(structure.copy())
                    if all(f(m) for f in filters):
                        yield m
                        break
            except ValueError:
                continue


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
class ElementScaledRattle(PerturbationABC):
    """Displace atoms by some amount from a normal distribution.

    Operates like :class:`.Rattle` but uses a standard deviation derived from the relative `sigma` and the `reference`,
    where this reference is given by element.
    """

    sigma: float
    reference: dict[str, float]
    create_supercells: bool = False
    "Create minimal 2x2x2 super cells when applied to structures of only one atom."

    def __call__(self, structure: Atoms):
        if self.create_supercells and len(structure) == 1:
            structure = structure.repeat(2)
        structure = super().__call__(structure)
        return element_scaled_rattle(structure, self.sigma, self.reference)

    def __str__(self):
        return f"scaled_rattle({self.sigma})"


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


__all__ = [
        "rattle",
        "stretch",
        "PerturbationABC",
        "Perturbation",
        "apply_perturbations",
        "Rattle",
        "Stretch",
        "Series",
        "RandomChoice",
]
