"""Generate plot_gallery.pkl — ~500 Cu-Zn structures with Morse energies."""
import pickle
from pathlib import Path

from ase.calculators.morse import MorsePotential
from assyst.crystals import Formulas, sample_space_groups
from assyst.relax import VolumeRelax, FullRelax, relax
from assyst.perturbations import apply_perturbations, Rattle, Stretch, RandomChoice
from assyst.filters import DistanceFilter, VolumeFilter, AspectFilter

# Cu-Zn binary formulas up to 6 atoms
forms = Formulas.range(("Cu", "Zn"), 6).trim(min_atoms=3, max_atoms=6)

structures = list(sample_space_groups(forms))
# subset just to keep data a bit smaller
structures = structures[::4]

# Attach Morse calculator (r0 ≈ Cu-Cu bond length, epsilon in eV)
calc = MorsePotential(r0=2.556, epsilon=0.336, rho0=6.0)


volset = VolumeRelax(max_steps=10, force_tolerance=1e-3)
allset = FullRelax(max_steps=100, force_tolerance=1e-3)

volmin = list(relax(volset, calc, structures))
allmin = list(relax(allset, calc, volmin))

rattle = Rattle(.25) + Stretch(hydro=.05, shear=0.005)
hydro = Stretch(hydro=.80, shear=.05)
shear = Stretch(hydro=.05, shear=.20)
stretch = RandomChoice(hydro, shear, .7)
mods = 1*[rattle] + 1*[stretch]

f = VolumeFilter(300) & DistanceFilter({'Cu': 1., 'Zn': 1.}) & AspectFilter(6)
rattle = list(apply_perturbations(mods, filters=f, structures=allmin))
structures = list(filter(f, structures + volmin + allmin + rattle))

for s in structures:
    s.calc = calc
    s.get_potential_energy()   # prime the calculator so energy is available post-pickle

out = Path(__file__).parent / "plot_gallery.pkl"
with open(out, "wb") as f:
    pickle.dump(structures, f)

print(f"Saved {len(structures)} structures to {out}")
