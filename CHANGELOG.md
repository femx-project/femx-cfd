# Changelog

## [Unreleased]

## [0.2.0] - 2026-07-18

### Added

- Added the `femx-forward` command for portable forward CFD runs.
- Added forward case format version 1.
- Added JSON progress reporting for GUI clients.
- Added status and result manifests.
- Added XDMF/HDF5 trajectory output.
- Added PETSc options and MPI execution support.
- Added Poisson inlet profiles on arbitrary inlet sections.
- Added automatic inlet-normal estimation.
- Added support for meshes declared in mm, cm, or m.

### Compatibility

- Requires `femx>=0.2.0,<0.3.0`.
- Compatible with femx-studio v0.2.x through forward case/result format version 1.

### Known limitations

- ReSolve is not yet available through the forward CFD workflow.