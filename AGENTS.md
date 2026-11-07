# AGENTS.md

Guidance for AI coding agents (Claude, Codex, Cursor, Aider, …) that **develop** assyst.
Guidance for *using* assyst lives in the docs, the notebooks and the agent skill
[`skills/assyst/SKILL.md`](skills/assyst/SKILL.md).

## What this is

`assyst` is the reference implementation of ASSYST (Automated Small SYmmetric Structure
Training). It generates symmetric crystal-structure training sets for machine-learning
interatomic potentials. It is built on ASE and does not depend on any particular MLIP,
reference method or workflow manager. It produces structures; it does not label them.
Paper: Poul, Huber, Neugebauer, *npj Comput. Mater.* 11, 174 (2025),
doi:10.1038/s41524-025-01669-4.

## Commands

```bash
pip install -e ".[test,plotneighborlist]"   # what CI installs (add `grace` for the tensorpotential tests)
pytest                                      # full suite + doctests in assyst/ (addopts = --doctest-modules,
                                            #   testpaths = tests, assyst); `pytest tests/` skips the doctests
pytest tests/test_crystal.py::test_foo      # one test;  pytest -k pattern  # by name
ruff check .                                # lint; line length 120 (pyproject). Not run in CI.
pip install -e ".[doc]" && make -C docs html   # Sphinx docs (furo + myst-nb)
```

- Python `>=3.11,<3.15`. CI runs the suite on 3.11–3.14.
- Extras: `test`, `coverage`, `doc`, `grace` (tensorpotential), `plotneighborlist`
  (matscipy) and `notebooks`.
- Tests that need tensorpotential are skipped when it is not installed
  (`tests/pickling/test_calculators.py`).

## Repo layout

| Path | What's there |
|------|--------------|
| `assyst/crystals.py` | `pyxtal()` wrapper, `Formulas` (composition algebra `+ \| *`, `range`, `trim`) and `sample()` (a generator over the (formula × space group) grid) |
| `assyst/relaxations.py` | `Relax` and its subclasses `CellRelax`, `VolumeRelax`, `SymmetryRelax`, `FullRelax`; `relax()`; `NotConvergedError` |
| `assyst/perturbations.py` | `rattle`/`stretch`/`element_scaled_rattle` functions; `PerturbationABC` and `Rattle`, `ElementScaledRattle`, `Stretch`, `Series`, `RandomChoice`; `perturb()` |
| `assyst/filters.py` | `FilterBase` (`&`/`\|`), `DistanceFilter` (+ `to_tol_matrix`), `AspectFilter`, `VolumeFilter`, `EnergyFilter`, `ForceFilter` |
| `assyst/calculators.py` | `AseCalculatorConfig` ABC (picklable, lazy calculator configs): `Grace` and `Morse` |
| `assyst/utils.py` | `update_uuid` (uuid/seed/lineage/step) and `step_of` |
| `assyst/neighbors.py` | `neighbor_list`: matscipy if installed, else ASE |
| `assyst/plot.py` | inspection plots for training sets (histograms, energy–volume, energy–distance) |
| `tests/` | pytest + hypothesis. Subdirs: `filter/`, `pickling/`, `reproducibility/`, `strategies/` (hypothesis strategies), `uuid/` |
| `notebooks/` | worked examples, committed **with executed outputs**: `SimpleUnary` (full pipeline), `ACEFitting`, `Relaxations`, `Perturbations`, `Lineage`, `PlotGallery`, `Quickstart/Crystals` |
| `docs/` | Sphinx sources. `conf.py` copies `notebooks/` into `docs/notebooks` at build time, with `nb_execution_mode = "off"` |
| `skills/assyst/` | agent skill for *users* of assyst (`SKILL.md` + `references/`); update it with any API change it describes |
| `paper/` | JOSS paper sources (`.github/workflows/draft-pdf.yml` builds the PDF) |

## Design rules

- **Stages are iterable-of-`Atoms` → iterable-of-`Atoms`.** Behaviour is configured by small
  `@dataclass(frozen=True)` objects. New behaviour is a new small class, not a new keyword
  argument on a top-level function. Filters and perturbations compose with operators
  (`&`, `|`, `+`, `RandomChoice`).
- **Every function that creates a structure must call `update_uuid(structure, step=...)`.**
  Lineage and `step` are part of the public contract (`docs/metadata.rst`,
  `tests/uuid/`). If you add an `info` key, document it in `docs/metadata.rst`.
- **Picklability.** Configuration objects must round-trip through pickle, because
  pipelines are shipped to remote workers (`tests/pickling/`). Calculators enter as an
  `AseCalculatorConfig` or as an ASE calculator on `Relax.calculator`.
- **Determinism.** Random classes take `rng: int | Generator | None` and normalise it with
  `RngMixin` (`np.random.default_rng`). The same seed must reproduce the same structures
  (`tests/reproducibility/`). Do not re-seed inside a loop: one generator should drive a
  whole grid (see the comment in `sample`).
- **Optional dependencies** are imported lazily. matscipy is wrapped in
  `pyiron_snippets.import_alarm.ImportAlarm` in `neighbors.py`, which warns and falls back
  to ASE. tensorpotential is imported inside `Grace.get_calculator()`, which raises
  `ImportError` with install instructions. `import assyst.*` must never need an extra.
- **Public names** go in each module's `__all__` and in `docs/api/<module>.rst`.
- ASE `Atoms` is the only currency between modules. Reference labels are supplied from
  outside. A `SinglePointCalculator` from a relaxation holds the relaxer's numbers, not
  labels.

## Working style

- Write in a terse, imperative tone in commits, PRs and comments. No marketing language.
- Use conventional commit subjects as in the history: `feat(relaxations): …`,
  `fix(filters): …`, `ci: …`, `build(deps-dev): …`, `plot: …`. Add the PR number on merge.
- Cite a commit, file path or command output for any claim. Every number comes with the
  script that produced it.
- One purpose per PR. Split notebooks, benchmarks and unrelated fixes into separate PRs.
- Tests assert tight conditions. A loose "it ran" test will be rejected. Hypothesis is
  welcome for property tests, and its example database is shared through CI artifacts
  (`tests/conftest.py`).
- **An API change updates every place that shows the old API**: docstrings, `docs/*.rst`
  (including the quick start in `docs/index.rst`) and the notebooks (re-executed, since
  the docs do not execute them).

## CI (`.github/workflows/`)

- `unittest.yml`: `pip install -e .[test,plotneighborlist]; pytest` on Python 3.11–3.14.
- `test-minimum-deps.yml`: `uv pip install --resolution lowest-direct` on Python 3.11.
  This exercises the declared lower bounds, so raise a floor when you use a newer feature.
- `codecov.yml`: coverage on 3.11.
- All three are gated by a path filter (`assyst/**`, `tests/**`, `pyproject.toml`, the
  workflow itself), so doc-only PRs skip them.
- `claude.yml` / `claude-code-review.yml`: the `@claude [haiku|sonnet|opus]` mention bot
  and automatic PR review.
- `dependabot.yml`: weekly pip updates (for example the `tensorpotential` cap).

## Releases

- The version comes from git tags via `setuptools-scm` (`assyst/_version.py` is generated
  and ignored by git). Tags are bare `X.Y.Z`; release candidates are `X.Y.Z-rcN`.
- Publishing a GitHub Release triggers `pypi-publish.yml`, which builds an **sdist only**
  and uploads it with PyPI trusted publishing.
- The conda-forge package (`assyst`) is updated through its feedstock. That happens
  outside this repo.
- Breaking API changes bump the major version (1.0.0 moved the calculator into
  `Relax.calculator`). Call them out in the release notes with a migration line.

## Citation

```
@article{poul2025automated,
    title = {Automated generation of structure datasets for machine learning potentials and alloys},
    journal = {npj Computational Materials},
    author = {Poul, Marvin and Huber, Liam and Neugebauer, Jörg},
    volume = {11}, number = {1}, pages = {174}, year = {2025},
    doi = {10.1038/s41524-025-01669-4},
}
```
