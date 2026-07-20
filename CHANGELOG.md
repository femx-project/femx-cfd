# Changelog

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
