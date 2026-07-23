# Case files

`femx-cfd` validates or runs one YAML case at a time. The file name is
arbitrary; `format`, `version`, and `method` select the workflow.

```shell
femx-cfd validate case.yaml
femx-cfd inspect case.yaml
femx-cfd run case.yaml
```

Paths inside a case are relative to the case file. An explicit `--output` path
is relative to the directory from which `femx-cfd` is run and overrides the
configured output directory.

## Formats

| Workflow | `format` | `version` | `method` |
| --- | --- | ---: | --- |
| Forward | `femx-cfd-forward` | 3 | — |
| Var | `femx-cfd-inverse` | 3 | `var` |
| EnVar | `femx-cfd-inverse` | 3 | `envar` |

Format versions are independent from the femx-cfd package version. Forward
versions 1 and 2 remain loadable and are migrated to version 3 automatically.
Inverse versions 1 and 2 also remain loadable; the version 1 normal-control
default is preserved during migration.

## Output

Version 3 cases can keep their result placement with the numerical settings:

```yaml
output:
  directory: runs/baseline
  every: 10
```

`directory` is relative to the case file and defaults to `result` beside that
file. `every` is a positive integer step interval and defaults to `1`.
It must divide the number of integration steps exactly, so the saved trajectory
has one fixed time interval and includes both the initial and final levels.
Inverse observation-time fields are interpolated separately from the complete
in-memory trajectory and do not add irregular levels to `flow_window.xdmf`.
This reduces stored trajectory data; it does not change the integration time
step or the states held in memory by the solver. Inverse result HDF5 version 3
records the selected indices and physical times as
`estimate/state_levels` and `estimate/state_times`.

Use `femx-cfd inspect case.yaml` for a read-only summary of the resolved output
directory, collision state, effective output interval and alignment, saved
level count, and inlet-based Reynolds and CFL estimates. A nonempty result
directory is never overwritten. An incompatible `every` value is rejected
before inspection or execution.

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

Current inverse cases estimate every inlet velocity component by default:

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
`estimate/parameters` and `estimate/coefficients`, together with parameter and
coefficient histories across outer iterations.

Only the final outer iteration's trajectory basis is saved by default. Retain
every outer basis when the intermediate linearizations are needed for
diagnostics:

```yaml
output:
  ensemble_basis: all
```

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

Set `ensemble.profile_scaling: peak` to scale these modes with one
time-independent peak inlet profile. The default `instantaneous` retains
relative perturbations proportional to the initial velocity at each time.

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
The synthetic recipe generates the mesh, truth trajectory, observations, and
reconstructed inlet; `var/case.yaml` and `envar/case.yaml` remain independent
runnable cases. Its `mesh` can either describe the built-in straight-tube
primitive or reference an existing Gmsh mesh:

```yaml
mesh:
  path: geometry/stenosis.msh
  length_unit: mm
```

Relative paths are resolved from `synthetic.yaml`. The source mesh must define
the physical groups referenced by the recipe. An existing meter-unit mesh is
used in place without being rewritten; generated meshes and converted
non-meter meshes are stored as `mesh/mesh.msh`. A file path and primitive-
generation options cannot be combined. Map every boundary condition explicitly
to its Gmsh physical-group name:

```yaml
boundary_conditions:
- boundary: inlet
  kind: velocity
  direction: auto
  profile:
    kind: poisson
    peak: 0.35
    skew: 0.0
  waveform:
    kind: harmonics
    mean: 1.0
    terms: []
- boundary: wall
  kind: velocity
  value: [0.0, 0.0]
- boundary: outlet
  kind: pressure
  value: 0.0
```

Exactly one velocity condition must contain `profile` and `waveform`; that
entry defines the analytic inlet. Other velocity and pressure entries are
uniform fixed conditions. Duplicate physical-group names are rejected. The
names and conditions are not inferred from geometry or hard-coded by the
synthetic generator. Solver settings are separate at the top-level `solver`
key because they are not boundary conditions.

Synthetic recipe version 2 accepts the same step interval:

```yaml
output:
  directory: generated
  every: 10
```

`directory` is resolved relative to `synthetic.yaml`; an explicit CLI
`--output` overrides it. Recipes without `directory` write beside the recipe.
Here, `every` must divide the integration steps
in one period. It writes the
uniformly sampled spin-up and observation trajectory as
`reference/flow_full.h5` and `reference/flow_full.xdmf`, the final
data-assimilation window as `reference/flow_window.h5` and
`reference/flow_window.xdmf` at the same interval, and exact fields at the
selected observation times as `reference/flow_observation.h5` and
`reference/flow_observation.xdmf`. It writes observation velocity data as
`observation/velocity.h5`, with `observation/velocity.pvd` and one VTU point
cloud per selected phase. The initial reconstructed inlet is
`inlet/initial_velocity.h5`. These visualization files make both the reference
field and sampled data directly inspectable in ParaView. Reference and
observation times use the inverse-analysis phase coordinate from zero through
one period; the full reference trajectory retains physical time. Observation
fields are interpolated from the unthinned solver trajectory, independently of
`output.every`. Use the same `every` in the synthetic recipe and its inverse
case when the reference and estimate window grids should match; `femx-cfd
inspect` reports the effective interval and number of saved levels for the
inverse case before it runs.
Within the analytic inlet boundary condition, `direction: auto` uses the inlet
plane's inward normal. An explicit two- or three-component `direction` is
normalized internally and can add a tangential component independently from
the spatial `profile`.
`observations.phases` defines the uniformly spaced phase grid. The optional
one-based `observations.selected_phases` list selects which phases are written
to the observation file. A single selected phase is extended as a constant
periodic vector field when constructing the initial inlet estimate.
Set `observations.mask: mesh` to create the Cartesian lattice over the
configured bounds and retain only points inside top-dimensional fluid mesh
elements. The default `none` preserves the complete rectangular lattice and
rejects the case later if any point is outside the mesh.
Synthetic reconstruction defaults to the existing `laplacian` projection.
Set `reconstruction.kind: poisson_fourier` with `depth` and `fourier_order` to
fit one mean-normalized Poisson inlet profile and a periodic Fourier waveform
over the inlet slab. `axial_decay` is optional for this method and weights
points nearer the inlet when provided.

## Results

Each run writes its case snapshot, status, and result manifest at the run root.
Inverse outputs use `estimate/result.h5` as their only HDF5 container;
`estimate/flow_window.xdmf` indexes the saved data-assimilation window and
`estimate/flow_observation.xdmf` selects its fields at observation phases.
`estimate/observation/velocity.pvd` visualizes predicted velocity at the same
points and phases as the observation input. The reconstructed inlet used to
start the analysis is `inlet/initial_velocity.h5`. A nonempty result directory
is never overwritten.

All HDF5 groups and datasets written by femx-cfd use lower-case snake-case
names. Forward and synthetic reference flow files use this structure:

```text
flow.h5
├── mesh
│   ├── geometry
│   └── topology
└── data
    └── step_000000
        ├── velocity
        └── pressure
```

Inverse result HDF5 version 3 uses the same convention:

```text
result.h5
├── mesh
│   ├── geometry
│   └── topology
├── estimate
│   ├── fields/step_000000
│   ├── observation_fields/step_000000
│   ├── parameters
│   ├── coefficients
│   ├── coefficient_names
│   ├── coefficient_history
│   └── coefficient_covariance
└── ensemble_basis
    └── outer_000
        ├── coefficient_names
        ├── state_levels
        ├── state_times
        ├── reference/fields
        └── perturbations/mode_000/fields
```

`ensemble_basis` is present for EnVar results. By default it stores the final
outer iteration only; set `output.ensemble_basis: all` to retain every outer
iteration. Each saved outer stores velocity and pressure for the reference
trajectory and difference fields for every coefficient mode at the same
uniformly saved times as the estimate. No additional CFD solves are performed
for this output. Its `ensemble_basis_outer_NNN.xdmf` view reconstructs each
physical member as `reference + perturbation` and also exposes the raw velocity
and pressure perturbations. It therefore includes the complete initial member
fields at time zero. The coefficient covariance is the local linear EnVar
covariance in coefficient space, formed from the weighted observation basis
and configured prior.
