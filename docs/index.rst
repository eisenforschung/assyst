.. assyst ASSYST documentation master file, created by
   sphinx-quickstart on Mon Jun 23 16:38:54 2025.

ASSYST documentation
====================

.. image:: https://zenodo.org/badge/997271420.svg
   :target: https://doi.org/10.5281/zenodo.15744358

.. image:: https://codecov.io/gh/pmrv/assyst/graph/badge.svg?token=NIEJ01UMJF
   :target: https://codecov.io/gh/pmrv/assyst

ASSYST is the Automated Small Symmetric Structure Training, a training protocol, aimed at providing comprehensive,
transferable training sets for machine learning interatomic potentials (MLIP) automatically. A detailed explanation and
verification of the method can be found in our papers. `[1]
<https://doi.org/10.1038/s41524-025-01669-4>`_ `[2] <https://doi.org/10.1103/PhysRevB.107.104103>`_ ASSYST gives up the
notion of fitting potentials to individual phases or structures and instead tries to deliver a
training set spanning the full potential energy surface (PES) of a material.

This software package is the reference implementation of this idea, designed to be as flexible as possible without
assuming either a specific MLIP, reference data, or workflow manager in mind.
It is built on `ASE <https://ase-lib.org/index.html>`_ and can use any of its calculators.
It also assumes that you label its output structures with reference energies and forces on your own, either with an ASE
calculator or by any other method.
For a ready-to-run implementation that targets Atomic Cluster Expansion and Moment Tensor Potentials fit to Density
Functional Theory (DFT) data check out `pyiron_potentialfit <https://github.com/pyiron/pyiron_potentialfit>`_.

Development happens on `Github <https://github.com/eisenforschung/assyst>`_, feel free to open any issues or pull
request for additional features.
We are open for any contributions!

Quick Start
-----------

Create a minimally viable training set in just 5 lines of python!

.. code-block:: python

   from assyst.crystals import Formulas, sample
   from assyst.relaxations import FullRelax, relax
   from assyst.perturbations import Rattle, perturb
   from assyst.calculators import Morse

   formulas = Formulas.range("Cu", 1, 5)
   structures = list(sample(formulas))

   calc = Morse()
   relaxed = list(relax(structures, FullRelax(), calc))
   training_set = list(perturb(relaxed, [Rattle(sigma=0.1)]))


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Contents:

   Home <self>
   installation
   background
   formulas
   filters
   notebooks

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Developer Notes:

   metadata
   custom_relaxer

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference:

   api/index
