# Portable cases

`femx-cfd` executes one versioned YAML case at a time:

```shell
femx-cfd validate case.yaml
femx-cfd run case.yaml --output result
```

The `format` field selects the workflow family. The `version` field versions
that family's configuration schema independently from the Python package.
The solver backend defaults to `petsc` when `solver.backend` is omitted.

## Forward cases

Forward case version 2 uses this identity:

```yaml
format: femx-cfd-forward
version: 2
```

It declares a mesh and its length unit, fluid properties, time integration,
one analytic inlet, fixed boundary conditions, and solver options. Paths are
resolved relative to the case file.

Fixed boundaries declare their mathematical condition explicitly:

```yaml
boundary_conditions:
  - boundary: wall
    kind: velocity
    value: [0.0, 0.0, 0.0]
  - boundary: outlet
    kind: pressure
    value: 0.0
```

Velocity vectors must match the mesh dimension. Forward version 1 cases remain
loadable; their `no_slip` boundaries are migrated to zero velocity vectors
before version 2 validation and execution.

## Inverse cases

Inverse case version 1 uses this identity:

```yaml
format: femx-cfd-inverse
version: 1
method: var
```

Supported methods are `var` and `envar`. An inverse case declares the shared
mesh, fluid, time, control boundary, fixed boundary conditions, prepared
velocity observations, inlet initialization, solver, and method options.
Its fixed boundaries use the same explicit velocity and pressure representation
as forward version 2.

Prepared inlet initialization references an HDF5 file:

```yaml
initialization:
  inlet:
    source: file
    path: initial_inlet.h5
```

The runner can instead derive it from the declared observations:

```yaml
initialization:
  inlet:
    source: observations
    projection:
      depth: 0.001
      axial_decay: 0.0005
      laplacian_modes: 6
      ridge: 0.001
```

The `femx-cfd example` subcommand selects one of the canonical complete cases
under the source repository's `examples` directory. These files are also
bundled with installed distributions:

```text
examples/
├── forward/straight_tube/case.yaml
└── inverse/straight_tube/
    ├── var.yaml
    └── envar.yaml
```

Regenerate the synthetic inverse inputs with:

```shell
python tools/build_synthetic_example.py --force
```

## Run output

Every run owns an initially empty result directory containing:

```text
result/
├── case.yaml
├── status.json
├── result.json
└── flow.xdmf
```

Forward results store their trajectory in `flow.h5`. Inverse results store
their trajectory, parameters, and metrics in `result.h5` and also snapshot the
prepared inlet as `initial_inlet.h5`.

`status.json` records `running`, `complete`, or `failed`. `result.json`
identifies the case family, method when applicable, and relative artifact
paths. Existing nonempty result directories are never overwritten.
