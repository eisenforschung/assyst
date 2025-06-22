'''Relaxation step of ASSYST.'''
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Iterable, Iterator

from .calculators import AseCalculatorConfig

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixSymmetry
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

import numpy as np

@dataclass(frozen=True, eq=True)
class Relax:
    '''Minimize energy with respect to internal positions.

    Also used as a base class for all other relaxation.'''
    max_steps: int = 100
    force_tolerance: float = 1e-3
    algorithm: Literal["LBFGS"] = "LBFGS"

    def apply_filter_and_constraints(self, structure: Atoms):
        '''Hook to allow subclasses to add filters and constraints.'''
        return structure

    def relax(self, structure: Atoms) -> Atoms:
        '''Relax a structure and return result.

        Structure must have a calculator attached.
        Returned structure will have a SinglePointCalculator with the final energy, forces and stresses attached.

        Args:
            structure (Atoms): structure to relax

        Returns:
            Atoms: relaxed structure with attached single point calculator.'''
        calc = structure.calc
        structure = structure.copy()
        structure.calc = calc
        lbfgs = LBFGS(self.apply_filter_and_constraints(structure), logfile="/dev/null")
        lbfgs.run(fmax=self.force_tolerance, steps=self.max_steps)
        calc = structure.calc
        structure.calc = SinglePointCalculator(
                structure,
                energy=calc.get_potential_energy(),
                forces=calc.get_forces(),
                stress=calc.get_stress()
        )
        relaxed_structure = structure
        relaxed_structure.constraints.clear()
        return relaxed_structure


@dataclass(frozen=True, eq=True)
class CellRelax(Relax):
    '''Minimize energy while keeping relative positions and volume constant.'''

    def apply_filter_and_constraints(self, structure: Atoms):
        structure.set_constraint(FixAtoms(np.ones(len(structure),dtype=bool)))
        return FrechetCellFilter(structure, constant_volume=True)


@dataclass(frozen=True, eq=True)
class VolumeRelax(Relax):
    '''Minimize energy while keeping relative positions and cell shape constant.'''
    pressure: float = 0.0

    def apply_filter_and_constraints(self, structure: Atoms):
        structure.set_constraint(FixAtoms(np.ones(len(structure),dtype=bool)))
        return FrechetCellFilter(structure, hydrostatic_strain=True, scalar_pressure=self.pressure)


@dataclass(frozen=True, eq=True)
class SymmetryRelax(Relax):
    ''' Minimize energy with respect to internal positions and cell, while keeping space group fixed.'''
    pressure: float = 0.0

    def apply_filter_and_constraints(self, structure: Atoms):
        structure.set_constraint(FixSymmetry(structure))
        return FrechetCellFilter(structure, scalar_pressure=self.pressure)


@dataclass(frozen=True, eq=True)
class FullRelax(Relax):
    ''' Minimize energy with respect to internal positions and cell without constraints.'''
    pressure: float = 0.0

    def apply_filter_and_constraints(self, structure: Atoms):
        return FrechetCellFilter(structure, scalar_pressure=self.pressure)


def relax(
        settings: Relax,
        calculator: AseCalculatorConfig | Calculator,
        structure: Iterable[Atoms]
) -> Atoms | Iterator[Atoms]:
    '''Relax structures according the given relaxation settings.

    Output structures have the final energy and force attached as ase's SinglePointCalculator.

    Args:
        settings (Relax): the kind of relaxation to perform (position, volume, etc.)
        calculator (AseCalculatorConfig or ase.calculator.Calculator): the energy/force engine to use
        structure (iterable of Atoms): the structures to minimize

    Yields:
        `Atoms` the corresponding relaxed configuration to each input structure
    '''
    for s in structure:
        if isinstance(calculator, AseCalculatorConfig):
            s.calc = calculator.get_calculator()
        else:
            s.calc = calculator
        yield settings.relax(s)
