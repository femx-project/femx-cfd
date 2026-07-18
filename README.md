# femx-cfd

Python applications for computational fluid dynamics (CFD) forward simulation,
data assimilation, and inverse analysis, built on
[femx](https://github.com/femx-project/femx).

## Setup

Clone the repository:

```shell
git clone https://github.com/femx-project/femx-cfd.git
cd femx-cfd
```

Create a Conda environment:

```shell
conda create -n femx-cfd python=3.12 pip
conda activate femx-cfd
conda install -c conda-forge "petsc>=3.19" hdf5 compilers cmake ninja pkg-config
```

Install femx and femx-cfd:

```shell
python -m pip install \
  "femx @ git+https://github.com/femx-project/femx.git@v0.2.0"
python -m pip install -e .
```

## Pulsatile-flow demo

```shell
femx-var-demo
femx-envar-demo
```
