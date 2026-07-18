# Changelog

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
