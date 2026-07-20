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

## Forward inlet modes

Set `inlet.source` to select how the forward inlet is defined:

| `source` | Required input | Behavior |
| --- | --- | --- |
| `analytic` (default) | `boundary`, `profile`; optional `normal`, `waveform`, `period` | Evaluates a theoretical spatial and temporal profile |
| `file` | `boundary`, `path`; optional `normal` | Reads inlet-node vector velocities from HDF5 and extracts the case normal component |

For example, read `inlet.h5` relative to the case file:

```yaml
inlet:
  source: file
  boundary: inlet
  path: inlet.h5
  normal: auto
```

The HDF5 file stores `velocity` with shape `(time, node, component)` and does
not store a normal. It must use the format written by `save_inlet`; version 1
normal-only files remain readable. Its boundary and time spacing must match the
case. Periodic data is repeated to `time.end_time`; nonperiodic data must cover
every simulation time step. See
[the forward example](../examples/forward/straight_tube/case.yaml) for a
complete analytic case.

## Inverse inputs

Inverse cases reference prepared velocity observations and initialize their
inlet from an HDF5 file or from those observations. Complete Var and EnVar cases
are available under `examples/inverse/`.

Observation-derived initialization reconstructs every velocity component on
the inlet. `projection.ridge` controls spatial smoothing, while
`projection.temporal_weight` controls time-derivative smoothing and defaults to
`0.0`.

## Examples

```shell
femx-cfd example forward
femx-cfd example var
femx-cfd example envar
```

Regenerate the synthetic inverse inputs with:

```shell
python tools/build_synthetic_example.py \
  --config examples/inverse/straight_tube/synthetic.yaml \
  --force
```

For a separate study, provide another config and output directory instead of
modifying the packaged example.
The synthetic recipe generates only the mesh, observations, and reconstructed
inlet; `var/case.yaml` and `envar/case.yaml` remain independent runnable cases.
Within `truth`, `direction: auto` uses the inlet plane's inward normal. An
explicit two- or three-component `direction` is normalized internally and can
add a tangential component independently from the spatial `profile`.

## Results

Each run writes its case snapshot, status, result manifest, and flow artifacts
to a new result directory. A nonempty result directory is never overwritten.
