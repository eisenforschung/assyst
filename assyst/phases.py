"""Utilities for handling phases and concentration jumps."""

from dataclasses import dataclass
from typing import Protocol, Iterable, List, Any


class Phase(Protocol):
    """Interface for a phase object."""

    def concentration(self, mu: float, T: float) -> float:
        """Returns the concentration at given chemical potential and temperature.

        Args:
            mu (float): Chemical potential.
            T (float): Temperature.

        Returns:
            float: Concentration.
        """
        ...


@dataclass(frozen=True)
class ConcentrationJump:
    """Represents a concentration jump found and refined.

    Attributes:
        c_left (float): Concentration on the left side of the jump.
        c_right (float): Concentration on the right side of the jump.
        mu (float): Chemical potential where the jump occurs.
        T (float): Temperature at which the jump was found.
    """

    c_left: float
    c_right: float
    mu: float
    T: float


def _get(obj: Any, attr: str) -> Any:
    """Helper to get attribute or key from object."""
    try:
        return getattr(obj, attr)
    except (AttributeError, TypeError):
        try:
            return obj[attr]
        except (KeyError, TypeError):
            raise ValueError(f"Object {obj} does not have attribute or key '{attr}'")


def refine_concentration_jumps(
    points: Iterable[Any],
    phase: Phase,
    jump_threshold: float = 0.01,
    mu_tolerance: float = 1e-5,
) -> List[ConcentrationJump]:
    """
    Checks for concentration jumps along isothermal mu lines and refines them.

    Args:
        points (Iterable): Iterable of points with attributes (or keys) c, mu, T.
        phase (Phase): Object with a concentration(mu, T) method.
        jump_threshold (float): Minimum concentration difference to be considered a jump.
        mu_tolerance (float): Desired precision for the refined mu value.

    Returns:
        List[ConcentrationJump]: List of refined concentration jumps.
    """
    # Group points by temperature
    isotherms: dict[float, List[Any]] = {}
    for p in points:
        t_val = float(_get(p, "T"))
        # Use a small tolerance for grouping temperatures
        found = False
        for existing_t in isotherms:
            if abs(existing_t - t_val) < 1e-6:
                isotherms[existing_t].append(p)
                found = True
                break
        if not found:
            isotherms[t_val] = [p]

    jumps = []
    # Sort temperatures for deterministic output
    for t_val in sorted(isotherms.keys()):
        isotherm_points = isotherms[t_val]
        # Sort by mu
        isotherm_points.sort(key=lambda p: _get(p, "mu"))

        for i in range(len(isotherm_points) - 1):
            p1 = isotherm_points[i]
            p2 = isotherm_points[i + 1]

            c1 = _get(p1, "c")
            c2 = _get(p2, "c")
            mu1 = _get(p1, "mu")
            mu2 = _get(p2, "mu")

            if abs(c2 - c1) > jump_threshold:
                # Refine the jump using bisection
                mu_low, mu_high = mu1, mu2
                c_low, c_high = c1, c2

                # We assume the jump is somewhere between mu_low and mu_high
                # and that the phase object will switch from c_low-like to c_high-like values.
                while mu_high - mu_low > mu_tolerance:
                    mu_mid = (mu_low + mu_high) / 2
                    c_mid = phase.concentration(mu_mid, t_val)

                    if abs(c_mid - c_low) < abs(c_mid - c_high):
                        mu_low = mu_mid
                        c_low = c_mid
                    else:
                        mu_high = mu_mid
                        c_high = c_mid

                jumps.append(
                    ConcentrationJump(
                        c_left=c_low,
                        c_right=c_high,
                        mu=(mu_low + mu_high) / 2,
                        T=t_val,
                    )
                )

    return jumps


__all__ = ["Phase", "ConcentrationJump", "refine_concentration_jumps"]
