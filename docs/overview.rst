Overview
========

ASSYST is the Automated Small Symmetric Structure Training, a training protocol, aimed at providing comprehensive,
transferable training sets for machine learning interatomic potentials automatically.
A detailed explanation and verification of the method can be found in two papers. `[1] <https://doi.org/10.1038/s41524-025-01669-4>`_ `[2] <https://doi.org/10.1103/PhysRevB.107.104103>`_
ASSYST gives up the notion of fitting potentials to individual phases or structures and instead tries to deliver a
training set spanning the full potential energy surface (PES) of a material.
The strategy to achieve this splits in three steps.

1. the exploration of the PES with randomly generated, but symmetric, periodic crystals in 
:ref:`assyst.crystals <crystals>`;

2. locating energetically favorable pockets in the PES by relaxing the initially generated sets of structures in
:ref:`assyst.relax <relax>`;

3. Exploring the direct neighborhood of these pockets by perturbing the relaxation configurations again in 
:ref:`assyst.perturbations <perturbations>`.

All three steps yield structures that are added to the training set.
Functions for each step are implemented in
