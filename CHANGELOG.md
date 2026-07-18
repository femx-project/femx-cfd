# Changelog

## v0.3.0 (unreleased)

### Added

- Added the `femx-cfd validate`, `femx-cfd run`, and `femx-cfd example`
  commands.
- Added portable inverse case format version 1 for Var and EnVar.
- Added common case loading, execution, progress, MPI, and result lifecycle
  support for forward and inverse workflows.
- Added self-contained forward, Var, and EnVar examples.
- Added a developer tool for regenerating the synthetic inverse example data.

### Changed

- Changed the development version to `0.3.0.dev0`.
- Moved application logic into `femx_cfd.cases`, `femx_cfd.workflows`, and
  `femx_cfd.runtime`.
- Changed `femx-var` and `femx-envar` to use portable inverse cases through the
  common runner.
- Preserved `femx-forward` as a compatibility alias for forward case version 1.
- Replaced Hydra/OmegaConf case loading with plain versioned YAML documents.
- Forwarded named backend options from portable forward cases to femx solvers.
- Moved synthetic mesh and observation generation helpers from the installed
  package to `tools/synthetic`.

### Removed

- Removed the installed top-level `apps` and `configs` packages.
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
