# Case files

`femx-cfd` validates or runs one YAML case at a time. The file name is
arbitrary; `format`, `version`, and `method` select the workflow.

```shell
femx-cfd validate case.yaml
femx-cfd run case.yaml --output result
```

Paths are relative to the case file.

## Formats

| Workflow | `format` | `version` | `method` |
| --- | --- | ---: | --- |
| Forward | `femx-cfd-forward` | 2 | — |
| Var | `femx-cfd-inverse` | 1 | `var` |
| EnVar | `femx-cfd-inverse` | 1 | `envar` |

Format versions are independent from the femx-cfd package version. Forward
version 1 remains loadable and is migrated to version 2 automatically.

## Fixed boundaries

Forward and inverse cases use explicit velocity vectors and pressure values:

```yaml
boundary_conditions:
  - boundary: wall
    kind: velocity
    value: [0.0, 0.0, 0.0]
  - boundary: outlet
    kind: pressure
    value: 0.0
```

Velocity vectors must match the mesh dimension. During forward version 1
migration, `no_slip` becomes a zero velocity vector.

## Inverse inputs

Inverse cases reference prepared velocity observations and initialize their
inlet from an HDF5 file or from those observations. Complete Var and EnVar cases
are available under `examples/inverse/`.

## Examples

```shell
femx-cfd example forward
femx-cfd example var
femx-cfd example envar
```

Regenerate the synthetic inverse inputs with:

```shell
python tools/build_synthetic_example.py --force
```

## Results

Each run writes its case snapshot, status, result manifest, and flow artifacts
to a new result directory. A nonempty result directory is never overwritten.
