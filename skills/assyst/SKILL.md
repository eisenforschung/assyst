---
name: assyst
description: >-
  Generate transferable training sets for machine-learning interatomic potentials
  (MLIPs) with assyst (Automated Small SYmmetric Structure Training): random symmetric
  crystals from pyxtal, then volume and full relaxations, then random rattles and
  strains, then filters. Read this BEFORE writing any assyst code, choosing
  perturbation amplitudes, relaxing seeds with a foundation model, or porting
  assyst 0.x code to 1.x. The relax() signature changed in 1.0, and several defaults
  silently drop or duplicate structures. Triggers: "assyst", "ASSYST",
  "training set generation", "random symmetric structures", "pyxtal seeds",
  "Formulas.range", "assyst.crystals.sample", "VolumeRelax", "FullRelax",
  "Rattle", "Stretch", "RandomChoice", "perturb", "DistanceFilter",
  "update_uuid", "lineage", "step_of", "relax_converged", "MLIP training data",
  "ACE / GRACE / MTP training set", "small cell training structures".
---

# assyst: automated small symmetric structure training sets

`assyst` is the reference implementation of the ASSYST protocol (Poul, Huber, Neugebauer,
*npj Comput. Mater.* **11**, 174 (2025), doi:10.1038/s41524-025-01669-4). It builds an MLIP
training set that spans the whole potential energy surface rather than one phase. It does
this in four steps, each a plain function from an iterable of `ase.Atoms` to another
iterable of `ase.Atoms`:

1. **sample**: random crystals in every space group, for each composition
   (`assyst.crystals`, uses pyxtal).
2. **relax**: a volume-only relaxation, then a full relaxation, to reach the local
   minima (`assyst.relaxations`, works with any ASE calculator).
3. **perturb**: random rattles and cell strains around each minimum
   (`assyst.perturbations`).
4. **filter**: drop unphysical structures (`assyst.filters`, which plug into the
   built-in `filter`).

assyst does **not** label structures. You compute the reference energies and forces
(DFT, a foundation model, …) yourself afterwards. It does not depend on any
particular MLIP, reference method or workflow manager.

- Source: <https://github.com/eisenforschung/assyst>. Docs: <https://assyst.readthedocs.io>.
- Install: `pip install assyst` or `conda install -c conda-forge assyst`. The optional
  extras are `assyst[grace]` (the GRACE foundation-model calculator config) and
  `assyst[plotneighborlist]` (matscipy, a faster neighbour list for filters and plots).
- Requires Python ≥ 3.11 and `pyxtal >=1.0.1,<1.2`.

## Check the version first

```python
import importlib.metadata as md; print(md.version("assyst"))
```

**1.0.0 (2026-10) broke the relaxation API.** Code written for 0.x will not run as is.
See `references/versions.md` for every difference.

| | 0.x (≤ 0.16) | ≥ 1.0 |
|---|---|---|
| calculator | `relax(structures, settings, calc)` | `relax(structures, VolumeRelax(calculator=calc))` |
| third positional arg of `relax` | `calculator` | `drop_unconverged: bool` |
| unconverged relaxations | returned silently, no flag | kept, `info["relax_converged"] = False`, plus a `UserWarning` |
| "which step made this?" | infer it from `len(info["lineage"])` | `assyst.utils.step_of(atoms)` (`"pyxtal"`, `"volume_relax"`, …) |
| `DistanceFilter(scalar)` | **broken: accepts everything** | works |
| `DistanceFilter` neighbour cutoff | fixed at 5 Å | adapts to the radii |

The 0.x call `relax(structures, FullRelax(), calc)` does not attach `calc` on 1.0: it
passes `calc` as `drop_unconverged`, and the relaxation raises
`ValueError: ... has no calculator` unless the structures already carry `.calc`.

## Core workflow (assyst ≥ 1.0)

```python
import numpy as np
from ase.calculators.emt import EMT            # stand-in: use your relaxation calculator
from assyst.crystals import Formulas, sample
from assyst.relaxations import VolumeRelax, FullRelax, relax
from assyst.perturbations import Rattle, Stretch, RandomChoice, perturb
from assyst.filters import DistanceFilter, AspectFilter, VolumeFilter
from assyst.utils import step_of

rng = np.random.default_rng(2025)               # ONE generator for the whole run

# 1. sample: Cu1..Cu4, all 230 space groups (incompatible pairs are skipped)
formulas = Formulas.range("Cu", 1, 5)
radii = {"Cu": 1.0}                             # per-element radius; a pair is rejected if d < r_i + r_j
seeds = list(filter(AspectFilter(6),           # larger cells: see pitfall 8 (RuntimeError ends the stream)
                    sample(formulas, max_atoms=4, tolerance=radii, rng=rng)))

# 2. relax: volume first, then everything
calc = EMT()
volmin = list(relax(seeds, VolumeRelax(calculator=calc, max_steps=100, force_tolerance=1e-3)))
fullmin = list(relax(volmin, FullRelax(calculator=calc, max_steps=100, force_tolerance=1e-3)))

# 3. perturb: the recipe from the upstream SimpleUnary notebook ("published settings")
rattle = Rattle(0.25, create_supercells=True, rng=rng) + Stretch(hydro=0.05, shear=0.005, rng=rng)
stretch = RandomChoice(Stretch(hydro=0.80, shear=0.05, rng=rng),   # choice_a: mostly hydrostatic
                       Stretch(hydro=0.05, shear=0.20, rng=rng),   # choice_b: mostly shear
                       0.7, rng=rng)                               # 0.7 = P(choice_b)
perturbed = list(perturb(fullmin, 4 * [rattle] + 4 * [stretch],
                         filters=[DistanceFilter(radii)]))

# 4. combine and filter. Every stage goes into the training set, not only the perturbed ones.
training = [s.copy() for s in filter(VolumeFilter(300), filter(DistanceFilter(radii),
            seeds + volmin + fullmin + perturbed))]
# .copy() drops the relaxer's SinglePointCalculator. Now label `training` with your reference method.
```

Notes on the code above:

- `sample` is a **generator** and `relax`/`perturb` are lazy, so `list(...)` them before
  reusing the results. `relax` and `perturb` copy their inputs and never mutate them.
- A `calculator` may be an ASE `Calculator`, or an `assyst.calculators.AseCalculatorConfig`
  such as `Grace(model=...)` or `Morse(...)`. A config is a small frozen dataclass that
  builds the calculator lazily, which keeps it **picklable**. Use a config whenever relaxations
  are sent to other processes or nodes, because ASE calculators usually do not pickle.
- If `calculator` is unset, `Relax` falls back to `structure.calc`.
- Relaxed structures carry a `SinglePointCalculator` with the **relaxer's** energies,
  forces and stress. If the relaxer is not your reference method, those numbers are not
  labels.

## Key parameters

| call | parameter | meaning / default |
|---|---|---|
| `Formulas.range(el, *range_args)` | like `range()` | `range("Cu", 3)` → Cu0, Cu1, Cu2 (the zero is kept). A list of elements gives the **outer product**. |
| `Formulas` algebra | `+` / `\|` / `*` | concatenate / zip (truncates to the shorter) / outer product. Operands must have disjoint elements. |
| `sample(...)` | `spacegroups` | default 1–230 (for `dim=3`) |
| | `min_atoms=1`, `max_atoms=10` | **formulas outside this window are dropped silently**, so raise `max_atoms` when you need bigger cells |
| | `tolerance` | `"metallic"` (default), `"atomic"`, `"molecular"`, `"vdW"`, a radii `dict`, or a `DistanceFilter` |
| | `shuffle=False` | `True` walks the (formula × group) grid in random order, so any prefix of the stream is spread across all formulas. Use it when you consume only the first N structures. |
| | `max_structures`, `rng`, `dim` | cap / seed or `Generator` / 0–3 for point, rod, layer and space groups |
| `Relax` subclasses | `max_steps=100`, `force_tolerance=1e-3` (eV/Å), `algorithm="LBFGS"\|"BFGS"\|"FIRE"`, `calculator` | `Relax` moves positions only. `CellRelax`: shape at fixed volume. `VolumeRelax`: isotropic scaling only. `SymmetryRelax`: everything, but keeps the space group. `FullRelax`: everything. |
| | `pressure=0.0` | **in eV/Å³** (the ASE convention, not GPa): `1 GPa = ase.units.GPa ≈ 0.00624 eV/Å³` |
| `relax(structures, settings, drop_unconverged=False)` | | ≥ 1.0 only |
| `Rattle(sigma, create_supercells=False, rng=None)` | `sigma` in Å | Gaussian standard deviation per Cartesian component |
| `ElementScaledRattle(sigma, reference, ...)` | `sigma` relative | multiplied by `reference[element]` (Å). Every element must be present in `reference`. |
| `Stretch(hydro, shear, minimum_strain=1e-3, rng=None)` | maxima | each diagonal and off-diagonal strain component is ±U(min(floor, max), max) |
| `RandomChoice(a, b, chance, rng)` | `chance` | probability of **b**, not of a |
| `perturb(structures, perturbations, filters=None, retries=10)` | | yields one structure per (input, perturbation) pair that passes the filters within `retries` |
| `DistanceFilter(radii)` | `dict[str, float]` or a scalar (≥ 1.0) | rejects if any pair has d < r_i + r_j (radii **summed**, so use roughly half a bond length). `.to_tol_matrix()` turns it into a pyxtal tolerance. |
| `AspectFilter(6)`, `VolumeFilter(max)` / `VolumeFilter(min, max)` | | longest/shortest lattice vector; Å³ per atom |
| `EnergyFilter(min_energy, max_energy)`, `ForceFilter(max_force)` | eV/atom; eV/Å | need a `SinglePointCalculator` on each structure, otherwise `ValueError` |
| filter algebra | `&`, `\|` | for `FilterBase` subclasses. Any `callable(Atoms) -> bool` also works. |

Metadata in `atoms.info`: `uuid`, `seed` (the uuid of the original crystal), `lineage` (the
uuids of all parents), `step` (≥ 1.0), `perturbation`, `relax_converged` (≥ 1.0),
`symmetry`/`requested spacegroup`, `spacegroup` (the actual one, which can be higher) and
`repeat`. A crystal from `sample()` carries the space-group keys; a crystal from a single
`pyxtal(group, species, num_ions)` call carries only `uuid`/`seed`/`step`.

## Pitfalls (each one has happened in real use)

1. **A fixed integer seed inside a per-seed function replays the same random draws for
   every seed.** `Stretch(..., rng=0)` turns the integer into a fresh
   `default_rng(0)` each time the object is built. In one campaign a helper built the
   perturbation objects with `rng=i` once per seed crystal. Across 120 seeds, perturbation
   *k* then had the same strain matrix: only 89 distinct strains among 711 perturbed
   structures. Build the perturbations once and share one `Generator`, or derive the seed
   from the crystal (`np.random.default_rng([base_seed, seed_index])`). Inference, not
   yet run: a process-pool or HPC executor pickles the perturbation objects, so every
   worker starts from the same RNG state and the same effect can appear.
2. **Perturbation amplitudes decide what the training set covers.** One campaign used
   amplitudes 5–40× below the upstream recipe (`Rattle` 0.1 Å, `Stretch(0.05, 0.005)`).
   Its design matrices were close to singular (condition numbers around 10¹⁵–10²⁸), and the
   fits extrapolated badly. The published-amplitude variant (±80 % hydrostatic, ±20 %
   shear) was the most transferable of all the methods compared, and the only one that
   kept diamond-Si stable. A training set certifies what its amplitudes cover and nothing
   beyond. Do not shrink them "to be safe".
3. **When `minimum_strain` exceeds a maximum, the floor is ignored and the clamp is degenerate.**
   For example, `Stretch(hydro=0.05, shear=0.005, minimum_strain=0.01)` draws
   U(min(0.01, 0.005), 0.005), so **every** shear component is exactly ±0.005. assyst only
   emits a `UserWarning`. Read your warnings, and report the strain distribution that was
   actually drawn rather than the one you configured.
4. **Single-atom cells are dropped silently from rattles.** `rattle()` raises `ValueError`
   on a 1-atom cell, and `perturb()` swallows every `ValueError`, including inside a
   `Rattle + Stretch` series. As a result, unary 1-atom minima get no rattled children.
   Use `Rattle(..., create_supercells=True)` (2×2×2 repeat), or count the output per
   perturbation type.
5. **Foundation-model references have holes.** When relaxing with a universal MLIP (GRACE,
   MACE, …), some relaxations fall into spurious deep minima and return fused cells with
   absurd energies. A fused structure poisons the training set and every perturbation
   built from it. DFT rarely fails this way, so the canonical recipe has no gate here. Add
   `filter(DistanceFilter(radii), ...)` after **each** relaxation. After labelling, add an
   energy-per-atom floor as a second, independent tripwire. **Count** every drop and report
   it. A silent filter hides a result.
6. **The default pyxtal tolerance can produce fused seeds for large atoms.**
   `tolerance="metallic"` allowed, for example, Yb pairs at about half the equilibrium
   spacing. Without relaxation those seeds were lost to the hole filter: in one campaign,
   43 % of structures and 38 of 120 seeds. Pass `tolerance=` the same radii dict (or
   `DistanceFilter`) that you filter with later. For the low-level `pyxtal()` wrapper use
   `tm=DistanceFilter(radii).to_tol_matrix()`, and optionally `factor=` (pyxtal's volume
   factor, passed through `**kwargs`).
7. **The radii dict must cover every element.** A missing element gets NaN, every
   comparison with it is False, and no pair involving it is ever rejected.
   `DistanceFilter({"Cu": 1.0})` accepts an Ag cell with 1.8 Å bonds. On ≤ 0.16 a
   **scalar** radius also accepted everything.
8. **Atom counts refer to pyxtal's conventional cell.** fcc (225) needs a multiple of 4
   atoms and bcc (229) a multiple of 2. With `Formulas.range("Cu", 1, 3)` most cubic groups
   are skipped silently (only 2 of the 6 pairs in {Cu1, Cu2} × {194, 225, 229} build). The
   (formula × group) grid is **finite**, and many pairs are incompatible. A
   one-at-a-time stream runs dry well below `len(formulas) * 230`. For more structures,
   loop over repeated passes with a continuing `Generator`.
   **pyxtal can raise `RuntimeError` ("long time to generate structure, check inputs"), and
   `sample()` does not catch it** (1.0.0 catches only `ValueError`). One hard (formula,
   group) pair then ends the whole stream with a traceback, and everything not yet
   `list()`ed is lost. It happens with tight tolerances and larger cells: Cu1..Cu10 with
   `tolerance={"Cu": 1.0}` failed after a few minutes, while Cu1..Cu4 did not. A tight
   radii tolerance is also slower (Cu1..Cu4: 110 s against 18 s for `"metallic"`). For
   large grids, call `sample` per pair with a shared `Generator` and skip failures:
   ```python
   seeds = []
   for f in formulas:
       for g in range(1, 231):
           try:
               seeds += sample([f], spacegroups=[g], max_atoms=n_max, tolerance=radii, rng=rng)
           except RuntimeError:
               pass                                   # count these and report them
   ```
9. **A structure needs a uuid before lineage means anything.** `sample()`/`pyxtal()`
   structures carry a `uuid`. Seeds made any other way (`ase.build.bulk`, a database)
   do not, and their first relaxation then starts the lineage from scratch, one level too
   shallow. Call `assyst.utils.update_uuid(atoms, step="my_source")` on foreign seeds. On
   ≥ 1.0, use `step_of()` to identify the stage instead of counting `lineage`.
10. **`RandomChoice(a, b, chance)` picks `b` with probability `chance`.** In the recipe
    above, 0.7 therefore favours the second argument (shear, measured at 0.697).
    Filters shift the mix further: `perturb()` redraws a rejected structure, so the choice
    that fails the filter more often is under-represented. In the core workflow above,
    on Cu with EMT, 222 hydrostatic against 1398 shear strains came out (14 % against the
    nominal 30 %), because ±80 % compressions often fail `DistanceFilter`. Count `step_of()`
    per perturbation.
11. **`EnergyFilter`/`ForceFilter(missing="ignore")` pass everything**, labelled structures
    included (verified on 0.16 and 1.0.0). Keep the default
    `missing="error"` and apply these filters only to labelled structures.
12. **ASSYST-style data has almost no short contacts.** In one 1.7 M-structure set, only 14
    structures had a pair below 1.5 Å. A fit from scratch never sees core repulsion and can
    fuse dimers. If short-range behaviour matters, add compressed or dimer structures, or a
    repulsive baseline. A related effect: the data are energetically narrow (74 % within
    2 eV/atom of the minimum), so set energy-weighting windows (for example pacemaker's
    `DElow`) from the measured spread, not from a published default.
13. **Upstream is `eisenforschung/assyst`.** Older personal forks are stale and lack
    modules such as `assyst.plot`.

## Verifying the output

Run these checks before labelling. Each is cheap.

```python
import collections
from assyst.utils import step_of
print(collections.Counter(step_of(s) for s in training))   # how many structures per stage
print(sum(not s.info.get("relax_converged", True) for s in fullmin), "unconverged of", len(fullmin))
print(len(seeds), "seeds;", len({s.info["seed"] for s in perturbed}), "seeds have perturbed children")
# expect ~8 per minimum; far fewer means a silent drop (1-atom rattles, filters)
print(len(perturbed) / len(fullmin), "perturbed per minimum")
# strain diversity: perturbation k must not repeat across seeds (pitfall 1)
```

- **Coverage plots** (`assyst.plot`): `volume_histogram`, `distance_histogram(rmax=6,
  reduce="min")`, `size_histogram`, `concentration_histogram`, `lattice_angle_histogram`.
  After labelling, also `energy_volume` and `energy_distance`. They take `list[Atoms]` and
  recompute neighbour lists on every call, so subsample very large sets first.
- **Minimum distance**: no structure should sit below your `DistanceFilter` floor. Look at
  the min-distance histogram for a cluster far below the bulk (fused cells).
- **Reproducibility**: run the pipeline twice with the same seed and compare positions and
  cells. `sample` with an integer or `Generator` `rng` is deterministic.
- **After labelling**: histogram the energy per atom (a far-left tail means holes) and the
  force components. Report every dropped structure by reason.

See `references/api.md` for signatures and `references/versions.md` for the 0.x → 1.x
differences.
