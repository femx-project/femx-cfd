# Case files

`femx-cfd` validates or runs one YAML case at a time. The file name is
arbitrary; `format`, `version`, and `method` select the workflow.

```shell
femx-cfd validate case.yaml
femx-cfd run case.yaml --output result
```

Paths inside a case are relative to the case file. An explicit `--output` path
is relative to the directory from which `femx-cfd` is run.

## Formats

| Workflow | `format` | `version` | `method` |
| --- | --- | ---: | --- |
| Forward | `femx-cfd-forward` | 2 | — |
| Var | `femx-cfd-inverse` | 2 | `var` |
| EnVar | `femx-cfd-inverse` | 2 | `envar` |

Format versions are independent from the femx-cfd package version. Forward
version 1 remains loadable and is migrated to version 2 automatically. Inverse
version 1 also remains loadable; its normal-control default is preserved during
migration to version 2.

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

Inverse version 2 estimates every inlet velocity component by default:

```yaml
control:
  boundary: inlet
  kind: vector
```

The `kind` field may be omitted because `vector` is the default. To estimate
only one scalar normal component instead, configure the optional normal mode:

```yaml
control:
  boundary: inlet
  kind: normal
  normal: auto
```

An explicit `control.normal` is valid only for normal control. Vector Var uses
one parameter per inlet node, time level, and mesh velocity component. Its
control bounds may be negative so tangential directions can have either sign.
For vector EnVar, every configured inlet mode is expanded once per velocity
component.

Velocity observations have equal weight by default. To emphasize selected
times, set one positive relative weight for every time level in the observation
HDF5 file. For example, the first time contributes four times the squared-error
weight of the other times here:

```yaml
observations:
  kind: velocity
  path: observation/velocity.h5
  time_weights: [4.0, 1.0, 1.0]
```

Internally, relative weight `w` is represented by an observation standard
deviation of `1 / sqrt(w)`. The list length is checked against the HDF5 time
count when the workflow starts.

Iterated EnVar re-centers the ensemble on the latest analysis and rebuilds the
spin-up and observation bases with the same configured modes. One outer
iteration preserves the single-linearization behavior:

```yaml
algorithm:
  prior_weight: 0.25
  outer_iterations: 2
```

`outer_iterations` defaults to `1`. The prior penalizes the local coefficient
increment in each outer iteration. Result HDF5 files retain the final
`param`/`coeff` and add `param_history`/`coeff_history` across outer iterations.
Set `ensemble.save_initial_states: true` to also write member-indexed XDMF
views containing the spin-up mean, every coupled initial field, and the
analysis.

EnVar modes combine a spatial pattern with a periodic temporal pattern. Define
the reusable patterns under `ensemble.modes`, then select only the combinations
that should be active:

```yaml
ensemble:
  modes:
    spatial:
      uniform: {kind: uniform}
      skew: {kind: surface_laplacian, index: 1}
    temporal:
      mean: {kind: constant}
      sin1: {kind: sine, order: 1}
    combined:
    - {name: flow_mean, spatial: uniform, temporal: mean, scale: 0.15}
    - {name: skew_sin1, spatial: skew, temporal: sin1, scale: 0.08}
```

`mode_scales` remains accepted for existing cases, but new cases should use
`modes`.

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
The synthetic recipe generates the mesh, complete truth trajectory,
observations, and reconstructed inlet; `var/case.yaml` and `envar/case.yaml`
remain independent runnable cases.
It writes the complete spin-up and observation trajectory as
`reference/flow_full.h5` and `reference/flow_full.xdmf`, the final
data-assimilation window as `reference/flow_window.h5` and
`reference/flow_window.xdmf`, and exact fields at the selected observation
times as `reference/flow_observation.h5` and
`reference/flow_observation.xdmf`. It writes observation velocity data as
`observation/velocity.h5`, with `observation/velocity.pvd` and one VTU point
cloud per selected phase. The initial reconstructed inlet is
`inlet/initial_velocity.h5`. These visualization files make both the reference
field and sampled data directly inspectable in ParaView. Reference and
observation times use the inverse-analysis phase coordinate from zero through
one period; the complete reference trajectory retains physical time.
Within the reference flow, `direction: auto` uses the inlet plane's inward normal. An
explicit two- or three-component `direction` is normalized internally and can
add a tangential component independently from the spatial `profile`.
`observations.phases` defines the uniformly spaced phase grid. The optional
one-based `observations.selected_phases` list selects which phases are written
to the observation file. A single selected phase is extended as a constant
periodic vector field when constructing the initial inlet estimate.

## Results

Each run writes its case snapshot, status, and result manifest at the run root.
Inverse outputs use `estimate/result.h5` as their only HDF5 container;
`estimate/flow_window.xdmf` indexes the complete data-assimilation window and
`estimate/flow_observation.xdmf` selects its fields at observation phases.
`estimate/observation/velocity.pvd` visualizes predicted velocity at the same
points and phases as the observation input. The reconstructed inlet used to
start the analysis is `inlet/initial_velocity.h5`. A nonempty result directory
is never overwritten.
