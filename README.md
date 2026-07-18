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

PETSc is the default solver backend.

## Examples

```shell
femx-cfd example forward
femx-cfd example var
femx-cfd example envar
```

Each command runs a bundled, self-contained example. Results are written below
`run/` by default, for example `run/var-example/`.
The canonical case files live under the source repository's `examples/`
directory and are bundled with installed distributions.

## Portable cases

Validate or run a case generated elsewhere:

```shell
femx-cfd validate path/to/case.yaml
femx-cfd run path/to/case.yaml --output path/to/result
```

Use `--progress json` for newline-delimited machine-readable progress.
`femx-forward` remains available as a compatibility alias for forward cases.

See [docs/cases.md](docs/cases.md) for the supported case families and output
contract.
