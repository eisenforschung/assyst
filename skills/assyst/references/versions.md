# assyst version differences that matter to users

Tags: `0.15.0` (2026-05-21), `0.16.0` (2026-07-30), `1.0.0-rc1`/`-rc2`, `1.0.0` (2026-10-22).
Read off `git diff 0.16.0 1.0.0 -- assyst/`.

## 0.16 → 1.0

| area | 0.16 | 1.0 | migrate |
|---|---|---|---|
| `relax()` | `relax(structures, settings, calculator)` | `relax(structures, settings, drop_unconverged=False)` | Move the calculator into the settings: `FullRelax(..., calculator=calc)`. A calc passed as the third positional argument becomes `drop_unconverged`, and the run then fails with `ValueError` (no calculator) unless `structure.calc` is set. |
| `Relax` fields | `max_steps`, `force_tolerance`, `algorithm` | adds `calculator` | `Relax.get_calculator()` resolves `calculator` first, then `structure.calc` |
| convergence | silent: an unconverged result looks like a minimum | `info["relax_converged"]`, a `UserWarning` with the count, `NotConvergedError`, and `drop_unconverged` | Check the flag before treating `fullmin` as minima. Custom `Relax.relax()` overrides should raise `NotConvergedError`. |
| step tracking | none | `info["step"]` (`pyxtal`, `relax`, `cell_relax`, `volume_relax`, `symmetry_relax`, `full_relax`, `<perturbation str>`) and `utils.step_of()` | Stop inferring the stage from `len(lineage)` |
| `DistanceFilter(scalar)` | a `defaultdict` whose `.get` never used the default, so **every structure passed** | a plain dict over all elements | Re-check any 0.x filtering that used a scalar radius |
| `DistanceFilter` cutoff | neighbour list fixed at 5 Å, so radius sums above 5 Å were not enforced | `2*max(r)+0.1` | n/a |
| `Formulas.range` docstring | claimed it skipped zero (it never did) | documents that zero is kept | n/a |
| `plot.distance_histogram` | broke with a reducer that returns an array | fixed | n/a |

## 0.15 → 0.16

- `sample(..., shuffle=False)` was added (PR #150).
- RNG handling in `sample` changed. In 0.15, `sample` passed `rng` unchanged to
  `pyxtal()` once per formula, so an **integer** seed re-seeded every formula with
  the same stream. In 0.16 one `Generator` drives the whole grid. As a result, 0.15 and
  0.16 produce different structures from the same integer seed.
