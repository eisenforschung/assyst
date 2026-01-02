.. assyst ASSYST documentation master file, created by
   sphinx-quickstart on Mon Jun 23 16:38:54 2025.

ASSYST documentation
====================

ASSYST is the Automated Small Symmetric Structure Training, a training protocol, aimed at providing comprehensive,
transferable training sets for machine learning interatomic potentials (MLIP) automatically. A detailed explanation and
verification of the method can be found in our papers. `[1]
<https://doi.org/10.1038/s41524-025-01669-4>`_ `[2] <https://doi.org/10.1103/PhysRevB.107.104103>`_ ASSYST gives up the
notion of fitting potentials to individual phases or structures and instead tries to deliver a
training set spanning the full potential energy surface (PES) of a material.

This software package is a minimal implementation of this idea, designed to be as flexible as possible without assuming
either a specific MLIP, reference data, or workflow manager in mind.
It is built on `ASE <https://ase-lib.org/index.html>`_ and can use any of its calculators.
It also assumes that you bring your own reference energies and forces.
For a ready-to-run implementation that targets Atomic Cluster Expansion and Moment Tensor Potentials fit to Density
Functional Theory (DFT) data check out `pyiron_potentialfit <https://github.com/pyiron/pyiron_potentialfit>`_.

Overview
========

The training strategy to achieve this splits in three steps.

1. the exploration of the PES with randomly generated, but symmetric, periodic crystals in 
:ref:`assyst.crystals <crystals>`;

2. locating energetically favorable pockets in the PES by relaxing the initially generated sets of structures in
:ref:`assyst.relax <relax>`;

3. Exploring the direct neighborhood of these pockets by perturbing the relaxation configurations again in
:ref:`assyst.perturbations <perturbations>`.

All three steps yield structures that are added to the training set.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Home <self>
   installation
   overview
   assyst
