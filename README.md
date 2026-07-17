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

## Forward simulation

`femx-forward` runs one portable YAML case. Mesh paths are resolved relative to
the case file, so Studio can place `case.yaml` and `mesh.msh` in the same job
directory. `mesh.length_unit` declares how the stored coordinates are
interpreted; the solver converts `mm` and `cm` meshes to SI meters before
assembly:

```yaml
format: femx-cfd-forward
version: 1
mesh:
  path: mesh.msh
  length_unit: mm

fluid:
  rho: 1050.0
  mu: 0.0035

time:
  dt: 0.001
  end_time: 0.1

inlet:
  boundary: inlet
  normal: auto
  profile:
    kind: poisson
    peak: 0.2
  waveform:
    kind: uniform
    value: 1.0

boundary_conditions:
  - boundary: wall
    kind: no_slip
  - boundary: outlet
    kind: pressure
    value: 0.0

solver:
  backend: petsc
  petsc_options: -ksp_type fgmres -pc_type bjacobi -sub_pc_type ilu
```

Run the case with:

```shell
femx-forward case.yaml
```

The default `result` directory contains the resolved `case.yaml`,
`status.json`, `result.json`, `flow.xdmf`, and `flow.h5`. GUI and job-runner
clients can request newline-delimited JSON progress on standard output:

```shell
femx-forward case.yaml --progress json
```

The case format and result manifest are the runtime interface used by
femx-studio. Studio launches this command in a separate Conda environment; it
does not import PETSc or femx-cfd into the Qt process.

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
