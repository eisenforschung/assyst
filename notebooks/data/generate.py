"""Generate plot_gallery.pkl — ~500 Cu-Zn structures with Morse energies."""
import pickle
from pathlib import Path

from ase.calculators.morse import MorsePotential
from assyst.crystals import Formulas, sample_space_groups

# Cu-Zn binary formulas up to 6 atoms
forms = Formulas.range(("Cu", "Zn"), 6).trim(max_atoms=6)

structures = list(sample_space_groups(forms, max_atoms=6))

# Attach Morse calculator (r0 ≈ Cu-Cu bond length, epsilon in eV)
calc = MorsePotential(r0=2.556, epsilon=0.336, rho0=6.0)
for s in structures:
    s.calc = calc
    s.get_potential_energy()   # prime the calculator so energy is available post-pickle

out = Path(__file__).parent / "plot_gallery.pkl"
with open(out, "wb") as f:
    pickle.dump(structures, f)

print(f"Saved {len(structures)} structures to {out}")
