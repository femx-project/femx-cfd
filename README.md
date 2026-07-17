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

Create and activate an isolated Conda environment:

```shell
conda create -n femx-cfd python=3.12 pip
conda activate femx-cfd
```

`femx` is a native extension. Its default Python build requires PETSc (with
MPI), HDF5, CMake, a C++ compiler, and `pkg-config`.

To use PETSc from Conda, install the complete native stack in the environment:

```shell
conda install -c conda-forge "petsc>=3.19" hdf5 compilers cmake ninja pkg-config
```

To use an existing PETSc installation:

```shell
conda install -c conda-forge cmake ninja pkg-config
export PETSC_DIR=/path/to/petsc
export HDF5_ROOT=/path/to/hdf5
```

Once either setup is complete, install the packages:

```shell
python -m pip install \
  "femx @ git+https://github.com/femx-project/femx.git@v0.2.0"
python -m pip install -e .
```

For development with a local `femx` checkout, install that checkout instead of
the tagged GitHub release:

```shell
python -m pip install -e /path/to/femx
python -m pip install -e .
```

ReSolve is optional. Enable it when building `femx` if its solver backends are
needed:

```shell
CMAKE_PREFIX_PATH=/path/to/resolve \
  python -m pip install \
  "femx @ git+https://github.com/femx-project/femx.git@v0.2.0" \
  -Ccmake.define.FEMX_ENABLE_RESOLVE=ON
```

## Pulsatile-flow demo

The demo assimilates sparse velocity observations in a rectangular channel to estimate its pulsatile inlet velocity. Run each stage separately:

```shell
# Generate the mesh.
femx-mesh --config-name demo/mesh

# Generate the analytic truth inlet.
femx-inlet --config-name demo/truth_inlet

# Run two truth cycles and sample observations.
femx-prepare --config-name demo/prepare

# Project the observations onto the inlet for the initial estimate.
femx-inlet --config-name demo/inlet

# Estimate the inlet with Variational method.
femx-var --config-name demo/var

# Estimate the inlet with the ensemble variational method.
femx-envar --config-name demo/envar
```

Or run the preparation stages followed by one analysis method:

```shell
femx-var-demo
femx-envar-demo
```
