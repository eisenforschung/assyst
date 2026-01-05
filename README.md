[![DOI](https://zenodo.org/badge/997271420.svg)](https://doi.org/10.5281/zenodo.15744358) [![Documentation Status](https://readthedocs.org/projects/assyst/badge/?version=stable)](https://assyst.readthedocs.io/stable/?badge=stable) [![codecov](https://codecov.io/gh/pmrv/assyst/graph/badge.svg?token=NIEJ01UMJF)](https://codecov.io/gh/pmrv/assyst)

# ASSYST or _Automated Small SYmmetric Structure Training_

A minimal reference implementation of ASSYST method to generate transferable training data for machine learning
potentials.

ASSYST is the Automated Small Symmetric Structure Training, a training protocol, aimed at providing comprehensive,
transferable training sets for machine learning interatomic potentials (MLIP) automatically. A detailed explanation and
verification of the method can be found in our papers.
[1](https://doi.org/10.1038/s41524-025-01669-4)[2](https://doi.org/10.1103/PhysRevB.107.104103) ASSYST gives up the notion of fitting potentials to
individual phases or structures and instead tries to deliver a training set spanning the full potential energy surface
(PES) of a material.

This software package is a minimal implementation of this idea, designed to be as flexible as possible without assuming
either a specific MLIP, reference data, or workflow manager in mind. It is built on
[ASE](https://ase-lib.org/index.html) and can use any of its calculators. It also assumes that you bring your own
reference energies and forces. For a ready-to-run implementation that targets Atomic Cluster Expansion and Moment Tensor
Potentials fit to Density Functional Theory (DFT) data check out pyiron_potentialfit.

![ASSYST schema](docs/img/AssystSchematic.svg)

## Citation

Please use the following citation when referencing the method in your work.

```
@article{poul2025automated,
  title={Automated generation of structure datasets for machine learning potentials and alloys},
  volume={11},
  DOI={10.1038/s41524-025-01669-4},
  number={1},
  journal={npj Computational Materials},
  author={Poul, Marvin and Huber, Liam and Neugebauer, J\"org},
  year={2025},
  month={Jun}
}
```
