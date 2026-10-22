---
title: 'ASSYST: A Python package for automated generation of training data for machine learning interatomic potentials'
tags: 
  - Python
  - Machine Learning
  - Potential
  - Machine Learning Interatomic Potentials
  - Physics
  - Materials Science
  - Atomistics
  - Simulation
authors:
  - name: Marvin Poul
    orcid: 0000-0002-6029-8748
    affiliation: 1
  - name: Jörg Neugebauer
    orcid: 0000-0002-7903-2472
    affiliation: 1
affiliations:
  - index: 1
    name: Max Planck Institute for Sustainable Materials, Germany
date: 21 May 2026
bibliography: paper.bib
---

# Summary

The accuracy of machine-learning interatomic potentials (MLIPs) depends
on high-quality training data, yet existing automation tools are often
monolithic and tied to specific electronic-structure codes. ASSYST is a
lightweight, pure Python library that implements the Automated Small
Symmetric Structure Training method. It has a modular design: small,
orthogonal classes communicate via abstract interfaces and the Atomic
Simulation Environment (ASE) Calculator API. The package can be easily
integrated with workflow managers, e.g., Executorlib [@executorlib] and
pyiron [@pyiron], command-line interfaces (CLI) or graphical user
interfaces (GUI), enabling heterogeneous high-performance computing (HPC)
deployment. Released under the BSD 3-clause license with full documentation and
CI testing, ASSYST enables the flexible and extensible generation of
MLIP training data across diverse research workflows.

# Statement of Need

Generation of training data is the leading expense in developing new
interatomic potentials, in terms of compute time as well as human
interaction. Various approaches have been developed to automate this
task via active learning, though the resulting code bases are often
specific to certain MLIPs and particular applications. This complicates
both the comparison of methods across models and systems and the
construction of foundational datasets that confer wide transferability
on their MLIPs.

Recently, we developed a method, the *Automated Small Symmetric Structure
Training* or *ASSYST*. It produces training sets consisting of large numbers of
small supercells, remaining agnostic to the MLIP architecture used to train on
the generated data while yielding exceptionally transferable MLIPs [@poul23;
@poul25]. Meanwhile, the field is rapidly advancing: universal MLIPs now model
significant parts of the periodic table [@grace]; extended formalisms
handle additional degrees of freedom beyond geometry [@yang2026macro]; and new
reference data expand the scope of potentials [@schran2019automated;
@herzog2024coupled]. All of these developments hint strongly that much more 
high-quality training data will be required in the near future, generated and
ingested by new data sources and new MLIP architectures that are difficult to
foresee. ASSYST therefore targets a design that facilitates interoperability
with different (electronic-structure) reference data, MLIPs, or workflow managers.

# State of the Field {#stateoffield}

There exist multiple software packages that aim to generate training
data for MLIPs (and often the MLIPs themselves, too), offering
integrated workflows to do so automatically or semi-automatically, e.g.,
[@gelzinyte23; @ZengDeepMD; @zhang2020dp; @autoplex, V.G. Fletcher [^2]] aimed at solid-state
materials. All of these packages are deeply integrated with the software
required to run the underlying electronic structure calculations and
MLIP fitting. While this can be convenient, it also makes setting up
additional MLIPs or electronic data generation methods very difficult.

By comparison, ASSYST aims to be more compact and easier to extend,
and to be less dependent on certain reference data or MLIPs. The
integration of ASSYST has been demonstrated in multiple public
workshops[^1]. The corresponding supporting material
illustrates the ease of integration with existing workflow tools, MLIP codes, and a
GUI---something that would be cumbersome to implement with other CLI-based
tools.

[^1]: See
[DPG2025](https://workshop.pyiron.org/DPG-tutorial-2025/02_assyst.html) and
[DPG2026](https://workshop.pyiron.org/DPG-tutorial-2026/03_assyst.html).

[^2]: V.G. Fletcher,
[NS_database_builder](https://github.com/VGFletcher/NS_database_builder)

# Software Design and Features

This package is the result of a refactoring of a previous implementation
of the method [@potentialfit]. The previous implementation followed an
integrated design pattern, similar to alternative packages mentioned
above. However, on attempting to integrate that previous code with
workflow tools and execution
models [@pyiron_core; @pyiron_workflow; @executorlib], we found it difficult
to cleanly express all options and capabilities of a monolithic code
in these smaller and more flexible contexts without a lot of code
duplication. Hence, we redesigned ASSYST from scratch by relying on:

1.  small, orthogonal classes, and

2.  clean, abstract interfaces between them.

The result is a lightweight code that can be quickly integrated into other
packages, even without in-depth knowledge of the code base. Users of
high performance computing (HPC) systems can benefit from this particularly
because the individual steps of a workflow producing an MLIP require quite
different and heterogeneous compute resources. Initial preparation tasks
may require minimal parallelization, if any, whereas electronic
structure calculations are almost always heavily optimized across
multiple compute nodes. Fitting the MLIP on the other hand is often best
done on specialized GPU-accelerated nodes. In a loosely coupled code
like ASSYST, users therefore have a lot of flexibility to submit
different parts of the workflow to the best-fitting HPC resources, e.g.,
with Executorlib [@executorlib].

This flexibility is also crucial to future-proof the package to changes
in MLIP architecture or reference data. ASSYST will require very few
or no changes to deal with other MLIPs or data, while more monolithic
designs, e.g., those discussed in the [State of the Field](#stateoffield)
section, are likely to require more modifications.

In \autoref{fig1classes}, we show the program design
that enables this flexibility. Green boxes depict individual workflow
steps. Between them data flows as lists of ASE  `Atoms`
objects [@ase-paper]. `Atoms` is a well-known and widely used data structure in
computational materials science that enables the easy distribution of
the individual workflow steps. In our previous GUI implementations[^1], each of
the boxes is represented by an interactive
'window' or 'node'. The ochre-colored boxes highlight possible and
common configuration options for each step, e.g., passing
`VolumeRelax("BFGS", pressure=0)` will instruct `relax(...)` to minimize
the energy of each structure at zero pressure using the well-known BFGS
algorithm. The violet boxes in turn mark where in the workflow ASSYST
relies on external codes to generate reference data or to act as a force
engine. These interactions occur either via the widely employed ASE
Calculator interface or by overloading a minimal set of documented methods,
which naturally integrate into the main workflow.

Finally, where ASSYST relies on random initialization or perturbation,
the state of the underlying random number generators is exposed to
enable fully reproducible runs. Each generated structure is assigned an
ID, and the relationships between derived structures are tracked. Together they
establish the provenance of each data point in a training set. This property is
verified in the test suite.

![Outline of data and control flow in the package. Green
boxes are the cleanly separated workflow steps. Arrows among them follow
the flow of generated structures. Ochre boxes present example code of the
small modular classes that are used to configure the individual workflow
steps. Violet boxes locate where ASSYST loosely couples to external
codes.\label{fig1classes}](Fig1_Features_Classes_InputTop_Shortlabel.png){ width=80% }

## GUI Example

We demonstrate the ease of integrating ASSYST with a small GUI for a partial
training set workflow in the listing below.
It uses `pyironflow` version 0.1.4 and `pyiron_workflow` version 0.20.3.
The modular structure allows integrators to wrap each workflow step separately
according to the requirements of the surrounding framework
This way exactly the wanted functionality can be exposed or small
incompatibilites glued over easily.
Because each GUI element cleanly maps onto a native workflow step, GUIs or
workflow managers can easily control the executation and persistance of the
steps.

\newpage

```python
import pyiron_workflow as pwf
from ase import Atoms
from pyironflow import PyironFlow

from assyst import crystals as cr
from assyst import perturbations as pt
from assyst.filters import DistanceFilter, FilterBase
from assyst.plot import volume_histogram

def formulas(element: str = "Cu", max_atoms: int = 4) -> cr.Formulas:
    formulas = cr.Formulas.range(element, 1, max_atoms + 1)
    return formulas

def sample(
    formulas: cr.Formulas, max_structures: int = 100, seed: int = 42
) -> list[Atoms]:
    structures = list(cr.sample(
        formulas, max_structures=max_structures,
        shuffle=True, rng=seed,
    ))
    return structures

def perturb(
    structures: list[Atoms],
    perturbation: pt.PerturbationABC,
    filters: FilterBase,
) -> list[Atoms]:
    structures = list(pt.perturb(
        structures, [perturbation], filters=filters
    ))
    return structures

def plot_volumes(structures: list[Atoms]):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    volume_histogram(structures)
    return fig

wf = pwf.Workflow("assyst")
wf.formulas = pwf.node(formulas)
wf.sample = pwf.node(sample, formulas=wf.formulas)
wf.rattle = pwf.node(
    pt.Rattle, sigma=0.1, create_supercells=True, rng=42
)
wf.min_dist = pwf.node(DistanceFilter, radii=0.8)
wf.perturb = pwf.node(
    perturb, structures=wf.sample, perturbation=wf.rattle,
    filters=wf.min_dist,
)
wf.volumes = pwf.node(plot_volumes, structures=wf.perturb)

pf = PyironFlow([wf], library_roots=None)
pf.gui
```

![Example of a GUI for part of an ASSYST
pipeline.  Each box maps directly to a native workflow step in
\autoref{fig1classes}.\label{fig2gui}](Fig2_Pyiron_Gui_Example.png)

# Research Impact Statement

ASSYST implements a method to generate training data that we described in
detail in [@poul23; @poul25].
While this implementation was not used in the original publications, those
papers prompted multiple independent investigations [@ito25; @bienvenue25;
@brunner26] using derived training sets.
The ASSYST package was created in part also to define a reference
implementation and facilitate future method extensions like these from a common
core.
ASSYST has been used in multiple public teaching workshops as well.

# Documentation and Source Code

Documentation including worked examples in notebooks is available at
<https://assyst.readthedocs.io/stable>. The source code can be obtained
under BSD 3-clause license at
<https://github.com/eisenforschung/assyst>.

# AI Usage Disclosure

Generative AI was used to generate the tests, parts of the
documentation, and a small number of features. The majority of AI
contributions are clearly marked as such by the commit information as
authored by `google-labs-jules[bot]` [@jules],
`Claude <noreply@anthropic.com>` or `claude[bot]`.
`jules` was used with Gemini 2.5 [@gemini_2_5] and 3 [@gemini_3_pro]; for
`Claude` we used Sonnet [@claude_sonnet_4_6] and Opus [@claude_opus_4_6;
@claude_opus_4_7; @claude_opus_5_5]. Prompts are linked in PR
descriptions where applicable. Additionally, most of the test suite was
AI-generated but committed manually in early commits, and is therefore not
marked. We ensured high quality by manual revision of all generated code and
documentation. These revisions were done openly on GitHub and can be
transparently reviewed in the corresponding pull request comments. The
architecture of the package has been entirely designed by us, and the clean and
orthogonal design supports the safe use of AI because it enables small
localized changes that are easily reviewed by human eyes.

# Acknowledgements

MP and JN acknowledge funding from the Deutsche Forschungsgemeinschaft
(DFG, German Research Foundation) through the Collaborative Research
Center 1394 (SFB 1394, No. 409476157).

# References
