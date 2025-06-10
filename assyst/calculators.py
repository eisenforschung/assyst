from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from functools import lru_cache
import os

from ase.calculators.calculator import Calculator
from ase.calculators.morse import MorsePotential

class AseCalculatorConfig(ABC):
    """Base class to keep calculator configurations.

    Exists mostly to avoid passing around potentially large and unpickle-able calculator objects."""
    @abstractmethod
    def get_calculator(self) -> Calculator:
        pass

@dataclass(frozen=True, eq=True)
class Grace(AseCalculatorConfig):
    """Universal Graph Atomic Cluster Expansion models."""
    model: str = "GRACE-FS-OAM"

    @lru_cache(maxsize=1)
    def get_calculator(self) -> Calculator:
        from tensorpotential.calculator import grace_fm
        # disable tensorflow warnings noise
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        return grace_fm(self.model)

@dataclass(frozen=True, eq=True)
class Morse(AseCalculatorConfig):
    """Morse potential for testing.  Parameters as in ASE."""
    epsilon: float = 1.0
    r0: float = 1.0
    rho0: float = 1.0

    def get_calculator(self) -> Calculator:
        return MorsePotential(**asdict(self))
