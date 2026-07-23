# Changelog

## Unreleased

### Added

- Added forward and inverse case version 3 with configured `output.directory`
  and uniformly spaced `output.every`, while retaining automatic migration
  from earlier case versions. The interval must divide the integration steps,
  and inverse observation fields are sampled separately from the full
  trajectory.
- Added `femx-cfd inspect` for read-only output collision, trajectory sampling,
  mesh, inlet Reynolds, and inlet CFL diagnostics.
- Added inverse result HDF5 version 3 with lower-case `mesh`, `estimate`, and
  `ensemble_basis` groups and explicit sampled state levels and times.
- Added full time-dependent EnVar reference and perturbation velocity/pressure
  fields for every outer iteration, reconstructing XDMF member views, named
  coefficients, coefficient histories, and local coefficient covariance.
- Added synthetic recipe version 2 with `output.every`; full and window
  reference trajectories remain uniform while observation fields are sampled
  independently from the unthinned truth trajectory.
- Added synthetic `output.directory`, resolved relative to the recipe and
  overridable from the generator CLI, matching portable CFD case placement.
- Added existing Gmsh mesh paths, with optional length-unit conversion, as an
  alternative to built-in primitive generation in synthetic recipes.
- Store generated synthetic meshes under `mesh/mesh.msh`; existing meter-unit
  mesh paths are reused in place without rewriting or copying them.
- Added mean-normalized Poisson spatial reconstruction with a fitted periodic
  Fourier waveform as an alternative synthetic inlet initialization.
- Added mutually exclusive `std` and linear `snr` specifications for synthetic
  Gaussian observation noise. SNR can use the observation RMS or maximum clean
  truth velocity, with resolved values recorded in the generated velocity HDF5
  metadata.
- Added explicit analytic and HDF5-file inlet modes to forward cases.
- Added workflow-independent inlet HDF5 version 2 with inlet-node vector
  velocity, case-owned normals, and version 1 read compatibility.
- Added three-component MRI inlet reconstruction with optional temporal
  regularization; its weight defaults to zero.
- Made synthetic mesh, truth, observation, and inlet reconstruction generation
  configurable through a versioned YAML document.
- Added an independently configurable, internally normalized synthetic truth
  direction; `auto` uses the inlet plane normal.
- Added inverse case version 2 with all-component inlet velocity control for
  Var and EnVar, while retaining optional normal-only control and version 1
  migration.
- Added configurable EnVar spatial, temporal, and combined inlet modes;
  each active mode is declared explicitly under `ensemble.modes`.
- Added configurable EnVar outer iterations that re-center the inlet ensemble,
  rebuild spin-up initial-state modes, and rebuild the observation basis.
- Added relative per-time weights for inverse velocity observations.
- Added selection of individual synthetic observation phases; a single phase
  can initialize a constant periodic vector inlet estimate.
- Added optional mesh masking for synthetic Cartesian observation lattices so
  points inside solid or out-of-domain regions are omitted before sampling.
- Added explicit boundary-condition mappings from synthetic recipes to Gmsh
  physical-group names. The analytic inlet entry owns its profile and waveform;
  recipe version 2 requires this structure explicitly.
- Made complete truth HDF5/XDMF and observation PVD/VTU visualization outputs
  mandatory for every generated synthetic case.
- Kept Var and EnVar analysis cases independent from synthetic-data recipes.
- Added observation-phase flow and point-velocity visualizations for every
  inverse result, alongside its saved estimated trajectory.

### Changed

- Save only the final EnVar outer iteration's trajectory basis by default;
  `output.ensemble_basis: all` retains every outer basis for diagnostics.
- Updated the supported femx dependency range to `femx>=0.4.0,<0.5.0`.
- Made PETSc synthetic generation MPI-safe: every rank participates in the
  truth solve, while rank zero alone writes shared mesh and result files; the
  process-owning generator CLI now finalizes PETSc before exit.
- Organized inverse examples as `var/case.yaml` and `envar/case.yaml`, with
  shared synthetic data in the parent case directory.
- Organized synthetic and inverse artifacts by their roles: `reference`,
  `observation`, `estimate`, and `inlet`. Inverse `result.h5` is now the sole
  HDF5 container under `estimate/`; XDMF/PVD files are views of its fields and
  predicted observations.
- Standardized every femx-cfd-owned HDF5 group and dataset on lower-case
  snake-case names, including `mesh/geometry`, `mesh/topology`,
  six-digit `data/step_000000`, and inverse field steps.

### Removed

- Removed the `femx-forward`, `femx-var`, and `femx-envar` command aliases;
  portable cases now use `femx-cfd run` exclusively.
- Removed the implicit EnVar `ensemble.mode_scales` translation; ensemble
  modes must be declared explicitly under `ensemble.modes`.
- Removed synthetic recipe version 1 migration after moving all maintained
  recipes to version 2.
- Removed the unused `waveform.harmonics` spelling; harmonic waveform entries
  use `waveform.terms`.
- Removed unused method-specific case loaders, output wrappers, MRI loading,
  and inlet-conversion wrappers from the internal Python surface.
- Removed the unused low-level compatibility-regularization branch that was
  not reachable from portable Var cases.

## v0.3.1

### Changed

- Updated the supported femx dependency range to `femx>=0.3.0,<0.4.0`.
- Updated solver timing expectations for femx v0.3.0, whose timing counters
  report the most recent solve.

## v0.3.0

### Added

- Added the `femx-cfd validate`, `femx-cfd run`, and `femx-cfd example`
  commands.
- Added portable inverse case format version 1 for Var and EnVar.
- Added forward case format version 2 with explicit fixed velocity and pressure
  conditions, plus automatic migration from forward case version 1.
- Added common case loading, execution, progress, MPI, and result lifecycle
  support for forward and inverse workflows.
- Added self-contained forward, Var, and EnVar examples.
- Added a developer tool for regenerating the synthetic inverse example data.

### Changed

- Moved application logic into `femx_cfd.cases`, `femx_cfd.workflows`, and
  `femx_cfd.runtime`.
- Moved the Var and EnVar implementations into `femx_cfd.da` while preserving
  their top-level `femx_cfd` imports.
- Changed `femx-var` and `femx-envar` to use portable inverse cases through the
  common runner.
- Preserved `femx-forward` as a compatibility alias for portable forward cases.
- Replaced Hydra/OmegaConf case loading with plain versioned YAML documents.
- Forwarded named backend options from portable forward cases to femx solvers.
- Moved synthetic mesh and observation generation helpers from the installed
  package to `tools/synthetic`.
- Moved canonical example sources to the repository-level `examples` directory
  and made CLI discovery manifest-driven.
- Changed bundled examples to select the PETSc solver backend explicitly.
- Changed current forward and inverse cases to represent fixed boundaries as
  explicit velocity vectors or pressure values.
- Kept PETSc reusable after Python API case runs while retaining process-owned
  finalization in CLI execution.

### Removed

- Removed the installed top-level `apps` and `configs` packages.
- Removed the `no_slip` preset from current case schemas and model workflows;
  forward version 1 migration retains compatibility with existing cases.
- Removed the redundant `femx_cfd.forward` compatibility module while
  preserving the forward API through top-level `femx_cfd` imports.
- Removed the public `femx-mesh`, `femx-prepare`, and `femx-inlet` pipeline
  commands; synthetic example construction now lives under `tools`.
- Removed the method-specific `femx-var-demo` and `femx-envar-demo` commands;
  bundled cases now run through `femx-cfd example var|envar`.

## v0.2.0 

### Added

- Added the `femx-forward` command for portable forward CFD runs.
- Added forward case format version 1.
- Added JSON progress reporting.
- Added status and result manifests.
- Added XDMF/HDF5 trajectory output.
- Added PETSc options and MPI execution support.
- Added Poisson inlet profiles on arbitrary inlet sections.
- Added automatic inlet-normal estimation.
- Added support for meshes declared in mm, cm, or m.

### Removed

- Removed the unused vector inlet-control path; prepared inlets now contain
  normal velocity only.

### Compatibility

- Requires `femx>=0.2.0,<0.3.0`.
