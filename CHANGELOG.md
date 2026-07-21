# Changelog

## Unreleased

### Added

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
  existing `mode_scales` settings remain supported.
- Added configurable EnVar outer iterations that re-center the inlet ensemble,
  rebuild spin-up initial-state modes, and rebuild the observation basis.
- Added optional member-indexed XDMF output for EnVar initial-state ensembles.
- Added relative per-time weights for inverse velocity observations.
- Added selection of individual synthetic observation phases; a single phase
  can initialize a constant periodic vector inlet estimate.
- Made complete truth HDF5/XDMF and observation PVD/VTU visualization outputs
  mandatory for every generated synthetic case.
- Kept Var and EnVar analysis cases independent from synthetic-data recipes.
- Added observation-phase flow and point-velocity visualizations for every
  inverse result, alongside its complete estimated trajectory.

### Changed

- Organized inverse examples as `var/case.yaml` and `envar/case.yaml`, with
  shared synthetic data in the parent case directory.
- Organized synthetic and inverse artifacts by their roles: `reference`,
  `observation`, `estimate`, and `inlet`. Inverse `result.h5` is now the sole
  HDF5 container under `estimate/`; XDMF/PVD files are views of its fields and
  predicted observations.

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
