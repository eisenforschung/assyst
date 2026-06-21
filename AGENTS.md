# AGENTS.md

Cross-tool entry point for AI coding agents (Codex, Cursor, Aider, Claude, ...).

## TL;DR

`assyst` — Automated Small SYmmetric Structure Training. Generates symmetric crystal-structure datasets for training ML interatomic potentials. Built on ASE; agnostic to MLIP, reference data, and workflow manager. Paper: [npj Comp. Mater. 11, 174 (2025)](https://doi.org/10.1038/s41524-025-01669-4).

## Commands

```bash
pip install -e ".[test]"                  # install with test deps
pytest tests/                             # full suite
pytest tests/test_crystal.py              # one file
pytest tests/test_crystal.py::test_foo    # one test
pytest -k pattern                         # by name
ruff check .                              # lint (line length 120, configured in pyproject.toml)
```

Python `>=3.11,<3.15`. Extras: `test`, `doc`, `coverage`, `grace` (tensorpotential), `plotneighborlist` (matscipy), `notebooks`.

Min-dep CI (`.github/workflows/test-minimum-deps.yml`, PR #140) installs every direct dep at its declared floor via `uv pip install --resolution lowest-direct` and runs the suite, so lower bounds are exercised on every PR.

## Repo layout

| Path | What's there |
|------|--------------|
| `assyst/crystals.py` | Crystal-structure generation (pyxtal-driven random symmetric structures) |
| `assyst/perturbations.py` | Random structure perturbations (`Perturbation` ABC + concretes) |
| `assyst/relaxations.py` | Relaxation step, ASE-calculator-driven |
| `assyst/calculators.py` | `AseCalculatorConfig` and friends |
| `assyst/filters.py` | `Filter` / `DistanceFilter` for structure rejection |
| `assyst/neighbors.py` | Neighbor-list helpers |
| `assyst/plot.py` | Plotting utilities |
| `assyst/utils.py` | Shared helpers (`update_uuid`, ...) |
| `tests/` | Pytest suite; subdirs for `filter/`, `pickling/`, `reproducibility/`, `strategies/`, `uuid/` |
| `notebooks/` | Usage examples |
| `docs/` | Sphinx sources |

## Conventions

- Frozen-ish `@dataclass` classes; ABC + concrete subclasses for `Perturbation`, `Filter`, etc.
- Optional deps gated through `pyiron_snippets.import_alarm.ImportAlarm` — importing without the extra raises only at instantiation, not import.
- ASE `Atoms` is the common currency between modules; reference labels (energies, forces) are supplied externally.
- Determinism matters: see `tests/reproducibility/` and `tests/uuid/` — random pipelines must round-trip through pickle and reproduce the same structure sequence.

## Working style

- Terse, imperative tone in commits, PRs, and comments. No marketing language.
- Cite a commit, file path, or command output for any claim. Numbers come with the script that produced them.
- One purpose per PR; split notebooks, benchmarks, and unrelated fixes off.
- Tests assert tight conditions; loose "it ran" tests get rejected.
- Notebooks are committed with executed outputs only.

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
