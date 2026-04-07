# Plot Gallery Notebook Design

**Date:** 2026-04-07
**Status:** Approved

## Overview

Create a self-contained reference notebook (`notebooks/PlotGallery.ipynb`) that demonstrates all 11 public plot functions in `assyst.plot`. The notebook loads a pre-generated, committed dataset and requires no external computation to run.

## Dataset Generation

**File:** `notebooks/data/generate.py`
**Output:** `notebooks/data/plot_gallery.pkl`

- Generate ~500 Cu-Zn binary structures using `assyst.crystals.sample_space_groups`
- Attach ASE Morse potential calculator to each structure and compute energies
- Pickle the list of `Atoms` objects to `notebooks/data/plot_gallery.pkl`
- Commit the pkl file to the repo so the notebook is immediately runnable

## Notebook Structure

**File:** `notebooks/PlotGallery.ipynb`

### Cell 1 — Title (markdown)
Brief description: "Reference notebook showing all available plot types in `assyst.plot`."

### Cell 2 — Setup
```python
import pickle
import matplotlib.pyplot as plt
from assyst.plot import (
    volume_histogram, size_histogram, concentration_histogram,
    distance_histogram, radial_distribution, energy_histogram,
    energy_distance, energy_volume,
    lattice_parameter_histogram, lattice_angle_histogram,
    aspect_ratio_histogram,
)

with open("data/plot_gallery.pkl", "rb") as f:
    structures = pickle.load(f)
```

### Cell 3 — Overview section (markdown)
Header: "Overview"

### Cell 4 — Overview grid (code)
One figure with all 11 plots using `plt.subplot(4, 3, i)` to direct each function call. All functions called with defaults only.

```python
fig = plt.figure(figsize=(15, 12))
plt.subplot(4, 3, 1);  volume_histogram(structures)
plt.subplot(4, 3, 2);  size_histogram(structures)
# ... etc for all 11
plt.tight_layout()
```

### Cells 5–N — Individual plots

One markdown header per category, one code cell per plot. Each cell calls the function directly with actively-used kwargs (no matplotlib boilerplate, no `plt.show()`). Kwargs are chosen to show meaningful effects.

#### Structural

**`volume_histogram`**
```python
volume_histogram(structures, bins=50, color="steelblue", density=True)
```
Shows: finer bins, custom color, normalized to density.

**`size_histogram`**
```python
size_histogram(structures, bins=20, color="tomato", rwidth=0.8)
```
Shows: custom color, bar width.

**`aspect_ratio_histogram`**
```python
aspect_ratio_histogram(structures, bins=40, color="mediumseagreen", histtype="step", linewidth=2)
```
Shows: step-style histogram, linewidth.

**`lattice_parameter_histogram`** (seaborn-backed)
```python
lattice_parameter_histogram(structures, bins=40, kde=True, alpha=0.5)
```
Shows: KDE overlay, transparency for overlapping histograms.

**`lattice_angle_histogram`** (seaborn-backed)
```python
lattice_angle_histogram(structures, bins=40, kde=True, alpha=0.5)
```
Shows: same pattern as lattice parameters.

#### Composition

**`concentration_histogram`**
```python
concentration_histogram(structures, elements=["Cu", "Zn"], width=0.03, color=["steelblue", "tomato"])
```
Shows: element filter, bar width, custom colors per element.

#### Distance / Neighbors

**`distance_histogram`**
```python
distance_histogram(structures, rmax=5.0, reduce="mean", bins=60, color="slateblue")
```
Shows: tighter cutoff, mean reduction instead of min.

**`radial_distribution`**
```python
radial_distribution(structures, rmax=7.0, bins=120, color="darkorange")
```
Shows: extended range, finer bins.

#### Energy

**`energy_histogram`**
```python
energy_histogram(structures, bins=60, color="crimson", density=True)
```
Shows: normalized density, custom color.

**`energy_distance`**
```python
energy_distance(structures, reduce="mean", rmax=5.0, alpha=0.4, color="steelblue")
```
Shows: mean distance reduction, alpha for overplotting.

**`energy_volume`**
```python
energy_volume(structures, alpha=0.4, color="mediumorchid")
```
Shows: scatter alpha.

## Key Decisions

- **No matplotlib boilerplate in individual cells** — functions write to `plt.gca()` directly; Jupyter displays inline automatically with `%matplotlib inline`.
- **Grid uses `plt.subplot` positioning** — functions use `plt.gca()` so `plt.subplot(n)` before each call is sufficient to redirect output.
- **Kwargs are active, not commented out** — every kwarg shown in individual cells produces a visible effect in the output.
- **Dataset committed to repo** — `plot_gallery.pkl` is checked in so the notebook runs without re-generating data.
