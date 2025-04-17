[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8010870.svg?style=flat-square)](https://doi.org/10.5281/zenodo.8010870)
[![PyPi Version](https://img.shields.io/pypi/v/dolfin-mech.svg?style=flat-square)](https://pypi.org/project/dolfin-mech)
[![PyPI Downloads](https://static.pepy.tech/badge/dolfin-mech)](https://pepy.tech/projects/dolfin-mech)

# dolfin_mech

A set of FEniCS-based python tools for Computational Mechanics.

The library has notably been used in:
* [[Genet (2019). A relaxed growth modeling framework for controlling growth-induced residual stresses. Clinical Biomechanics.](https://doi.org/10.1016/j.clinbiomech.2019.08.015)]
* [[Álvarez-Barrientos, Hurtado & Genet (2021). Pressure-driven micro-poro-mechanics: A variational framework for modeling the response of porous materials. International Journal of Engineering Science.](https://doi.org/10.1016/j.ijengsci.2021.103586)]
* [[Patte, Genet & Chapelle (2022). A quasi-static poromechanical model of the lungs. Biomechanics and Modeling in Mechanobiology.](https://doi.org/10.1007/s10237-021-01547-0)]
* [[Patte, Brillet, Fetita, Gille, Bernaudin, Nunes, Chapelle & Genet (2022). Estimation of regional pulmonary compliance in idiopathic pulmonary fibrosis based on personalized lung poromechanical modeling. Journal of Biomechanical Engineering.](https://doi.org/10.1115/1.4054106)]
* [[Tueni, Allain & Genet (2023). On the structural origin of the anisotropy in the myocardium: Multiscale modeling and analysis. Journal of the Mechanical Behavior of Biomedical Materials.](https://doi.org/10.1016/j.jmbbm.2022.105600)]
* [[Laville, Fetita, Gille, Brillet, Nunes, Bernaudin & Genet (2023). Comparison of optimization parametrizations for regional lung compliance estimation using personalized pulmonary poromechanical modeling. Biomechanics and Modeling in Mechanobiology.](https://doi.org/10.1007/s10237-023-01691-9)]
* [[Peyraut & Genet (2024). A model of mechanical loading of the lungs including gravity and a balancing heterogeneous pleural pressure. Biomechanics and Modeling in Mechanobiology.](https://doi.org/10.1007/s10237-024-01876-w)]
* [[Peyraut & Genet (2024). Finite strain formulation of the discrete equilibrium gap principle: application to direct parameter estimation from large full-fields measurements. Comptes Rendus Mécanique.](https://doi.org/10.5802/crmeca.279)]

### Installation

A working installation of [FEniCS](https://fenicsproject.org) (version 2019.1.0) is required to run `dolfin_mech`.
To setup a system, the simplest is to use [conda](https://conda.io): first install [miniconda](https://docs.conda.io/projects/miniconda/en/latest) (note that for Microsoft Windows machines you first need to install WSL, the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install), and then install miniconda for linux inside the WSL; for Apple MacOS machines with Apple Silicon CPUs, you still need to install the MacOS Intel x86_64 version of miniconda), and then install the necessary packages:
```
conda create -y -c conda-forge -n dolfin_mech fenics=2019.1.0 matplotlib=3.5 meshio=5.3 mpi4py=3.1.3 numpy=1.24 pandas=1.3 pip python=3.10 vtk=9.2
conda activate dolfin_mech
```
