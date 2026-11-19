# assyst API reference (as of 1.0.0)

Signatures copied from the source at tag `1.0.0` (commit `afcf858`, released 2026-10-22).
Each module defines `__all__`; only the names exported there are listed.

## `assyst.crystals`

```python
pyxtal(group: int | list[int], species: tuple[str], num_ions: tuple[int],
       dim: Literal[0,1,2,3] = 3, repeat: int = 1, allow_exceptions: bool = True,
       rng: int | np.random.Generator | None = None, **kwargs) -> Atoms | list[dict]
```
- With one `int` group and `repeat == 1`, it returns a single `Atoms`. It raises `ValueError`
  if the stoichiometry is incompatible with the group.
- Otherwise it returns `list[dict]` with the keys `atoms`, `symmetry`, `repeat`,
  `requested spacegroup` and `spacegroup`. Incompatible groups are skipped with one
  `UserWarning`.
- `**kwargs` go straight to `pyxtal.pyxtal.from_random`. Useful ones are
  `tm=Tol_matrix` (minimum distances) and `factor=` (volume factor).
- pyxtal itself may raise `RuntimeError` when it cannot place atoms in time. Neither this
  wrapper nor `sample()` catches it, so in `sample()` it ends the whole stream (SKILL.md, pitfall 8).

```python
@dataclass(frozen=True) class Formulas(Sequence):  atoms: tuple[dict[str, int], ...]
Formulas.range(elements: str | Iterable[str], *range_args) -> Formulas   # builtin-range semantics, keeps 0
Formulas.trim(min_atoms=1, max_atoms=None) -> Formulas
Formulas.elements -> set[str]
+  concatenate      |  zip (truncate to shorter)      *  outer product   (| and * need disjoint elements)
```

```python
sample(formulas, spacegroups=None, min_atoms=1, max_atoms=10, max_structures=None,
       dim=3, tolerance="metallic", shuffle=False, rng=None) -> Iterator[Atoms]
```
- `tolerance` is one of `"metallic" | "atomic" | "molecular" | "vdW"`, a `dict` of
  element radii, or a `DistanceFilter`. An empty dict means no tolerance.
- Formulas are trimmed to `[min_atoms, max_atoms]`. Zero counts are dropped, and the empty
  formula is skipped.
- One `Generator` drives the whole grid, so the result is deterministic for a given `rng`.
  `shuffle=True` permutes the (formula, group) grid with the same generator.
- `sample` (from 0.16) adds the info keys `symmetry`, `repeat`, `requested spacegroup` and
  `spacegroup` (the actual group, found by re-analysing with pyxtal).

## `assyst.relaxations`

```python
class NotConvergedError(RuntimeError): structure: Atoms

@dataclass(frozen=True, eq=True)
class Relax:                       # positions only
    max_steps: int = 100
    force_tolerance: float = 1e-3      # eV/Å, passed to optimizer.run(fmax=...)
    algorithm: Literal["LBFGS", "BFGS", "FIRE"] = "LBFGS"
    calculator: AseCalculatorConfig | Calculator | None = None
    def get_calculator(self, structure) -> Calculator   # self.calculator, else structure.calc, else ValueError
    def __call__(self, structure) -> Atoms               # never raises NotConvergedError; sets info["relax_converged"]
    def relax(self, structure) -> Atoms                  # override point for custom engines; raises NotConvergedError
    def apply_filter_and_constraints(self, structure)    # hook for subclasses

CellRelax(Relax)                     # cell shape at constant volume, positions fixed
VolumeRelax(Relax, pressure=0.0)     # hydrostatic only, positions fixed
SymmetryRelax(Relax, pressure=0.0)   # FixSymmetry + FrechetCellFilter
FullRelax(Relax, pressure=0.0)       # FrechetCellFilter, unconstrained
# pressure is in eV/Å³ (ASE scalar_pressure)

relax(structures, settings: Relax, drop_unconverged: bool = False) -> Iterator[Atoms]
```
- The output is a copy of each input, with a new uuid, `step = str(settings)` (for example
  `"full_relax"` or `"volume_relax(pressure=0.1)"`) and a `SinglePointCalculator` holding
  the final energy, forces and stress. Constraints are cleared.
- After the loop, `relax()` emits one `UserWarning` that counts the unconverged structures.
- A custom engine subclasses `Relax`, overrides `relax()`, calls `update_uuid`, attaches a
  `SinglePointCalculator`, and raises `NotConvergedError(structure)` when it does not
  converge. See `docs/custom_relaxer.rst`.

## `assyst.perturbations`

```python
rattle(structure, sigma, rng=None)                      # in place; ValueError on 1-atom cells
element_scaled_rattle(structure, sigma, reference, rng=None)
stretch(structure, hydro, shear, minimum_strain=1e-3, rng=None)   # in place; cell @ (I + eps), symmetric eps

Rattle(sigma, create_supercells=False, rng=None)
ElementScaledRattle(sigma, reference: dict[str, float], create_supercells=False, rng=None)
Stretch(hydro, shear, minimum_strain=1e-3, rng=None)
RandomChoice(choice_a, choice_b, chance, rng=None)      # P(choice_b) = chance
Series(perturbations)                                   # also built by a + b
perturb(structures, perturbations, filters=None, retries=10) -> Iterator[Atoms]
```
- Every `PerturbationABC` call runs `update_uuid(structure, step=str(self))` and appends
  `str(self)` to `info["perturbation"]`, joined with `+`.
- `rng` goes through `np.random.default_rng()` in `__post_init__`. An `int` therefore
  becomes a **new** generator for each object, while a `Generator` is shared as is.
- `perturb` copies each input, retries a perturbation until all filters pass (at most
  `retries` times), and **silently skips** any perturbation that raises `ValueError`.
- A plain `callable(Atoms) -> Atoms` is accepted wherever a perturbation is expected. It is
  wrapped in `FunctionPerturbation`.

## `assyst.filters`

```python
DistanceFilter(radii: dict[str, float] | float)   # reject if any d_ij < r_i + r_j; NaN radius = always allowed
    .to_tol_matrix(prototype="metallic") -> pyxtal Tol_matrix
AspectFilter(maximum_aspect_ratio=6)               # longest / shortest cell vector
VolumeFilter(max) | VolumeFilter(min, max) | VolumeFilter(minimum_volume_per_atom=..., maximum_volume_per_atom=...)
EnergyFilter(min_energy=-inf, max_energy=+inf, *, missing="error")   # eV/atom
ForceFilter(max_force=+inf, *, missing="error")                      # eV/Å, max over atoms of |F|
AndFilter(l, r), OrFilter(l, r)                    # or FilterBase & / |
```
- Each filter is a predicate: `True` keeps the structure. Use them with the built-in
  `filter(f, structures)` or with `perturb(..., filters=...)`.
- `DistanceFilter` uses a neighbour-list cutoff of `2 * max(radii, 2.5) + 0.1` Å (≥ 1.0).
  `neighbor_list` comes from matscipy if it is installed, else from ASE (with a warning
  from `pyiron_snippets.ImportAlarm`).
- `missing="ignore"` on `EnergyFilter`/`ForceFilter` skips the check for **every**
  structure, not only those that lack a calculator. This is an upstream bug; see NOTES.

## `assyst.calculators`

```python
AseCalculatorConfig (ABC): get_calculator() -> Calculator
Grace(model="GRACE-FS-OAM")      # needs tensorpotential; lru_cache'd per instance; sets TF_CPP_MIN_LOG_LEVEL=3
Morse(epsilon=1.0, r0=1.0, rho0=1.0)
```

## `assyst.utils`

```python
update_uuid(structure, step: str | None = None) -> Atoms   # push old uuid onto lineage, new uuid, set seed if absent
step_of(structure, default="unknown") -> str                # ≥ 1.0
STEP_KEY = "step"
```

## `assyst.plot`

All of these functions take `list[Atoms]` plus `**kwargs` for matplotlib or seaborn:
`volume_histogram`, `size_histogram`, `concentration_histogram(elements=None)`,
`distance_histogram(rmax=6.0, reduce="min"|"mean"|callable|None)`,
`radial_distribution(rmax=6.0)`, `energy_histogram`, `energy_distance(rmax, reduce)`,
`energy_volume`, `lattice_parameter_histogram`, `lattice_angle_histogram` and
`aspect_ratio_histogram`.

- The energy plots need a `SinglePointCalculator` on each structure.
- `energy_*` switch from a scatter plot to a log-binned hexbin above about 1000 points.
- Multi-series histograms share one bin grid (seaborn `histplot`, `element="step"`).
