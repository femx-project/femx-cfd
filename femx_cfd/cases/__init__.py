"""Load and validate versioned portable CFD case files."""

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path

import yaml

from femx_cfd._typing import Config, PathLike

from .forward import CASE_FORMAT as FORWARD_FORMAT
from .forward import CASE_VERSION as FORWARD_VERSION
from .forward import validate_forward_case
from .inverse import CASE_FORMAT as INVERSE_FORMAT
from .inverse import CASE_VERSION as INVERSE_VERSION
from .inverse import validate_inverse_case


@dataclass(frozen=True)
class LoadedCase:
    """One source case document and its path-resolved runtime form."""

    path: Path
    document: Config
    resolved: Config

    @property
    def format(self) -> str:
        return self.resolved["format"]

    @property
    def version(self) -> int:
        return self.resolved["version"]

    @property
    def source_version(self) -> int:
        """Schema version declared by the source document."""

        return self.document["version"]


def load_case(path: PathLike) -> LoadedCase:
    """Load a supported case and dispatch validation by its format."""

    case_path = Path(path).expanduser().resolve()
    if not case_path.is_file():
        raise ValueError(f"case does not exist: {case_path}")
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("case root must be a mapping")

    case_format = document.get("format")
    if case_format == FORWARD_FORMAT:
        resolved = validate_forward_case(document, base_dir=case_path.parent)
    elif case_format == INVERSE_FORMAT:
        resolved = validate_inverse_case(document, base_dir=case_path.parent)
    else:
        raise ValueError(f"unsupported case format: {case_format!r}")

    return LoadedCase(
        path=case_path,
        document=deepcopy(document),
        resolved=resolved,
    )


def validate_case(path: PathLike) -> LoadedCase:
    """Validate a case file and return its loaded representation."""

    return load_case(path)


def snapshot_case(case: LoadedCase, directory: PathLike) -> Config:
    """Return an auditable case snapshot with paths valid from a result."""

    document = deepcopy(case.document)
    target = Path(directory).expanduser().resolve()
    if case.format == FORWARD_FORMAT:
        mesh = Path(case.resolved["mesh"]["path"])
        relative = os.path.relpath(mesh, target)
        mesh_options = document.get("mesh")
        if isinstance(mesh_options, str):
            document["mesh"] = relative
        else:
            document["mesh"]["path"] = relative
        inlet = case.resolved["inlet"]
        if inlet["source"] == "file":
            document["inlet"]["path"] = os.path.relpath(inlet["path"], target)
    elif case.format == INVERSE_FORMAT:
        paths = (
            ("mesh", "path", case.resolved["mesh"]["path"]),
            (
                "observations",
                "path",
                case.resolved["observations"]["path"],
            ),
        )
        for section, name, source in paths:
            document[section][name] = os.path.relpath(source, target)
        inlet = case.resolved["initialization"]["inlet"]
        if inlet["source"] == "file":
            document["initialization"]["inlet"]["path"] = os.path.relpath(
                inlet["path"], target
            )
    return document


__all__ = [
    "FORWARD_FORMAT",
    "FORWARD_VERSION",
    "INVERSE_FORMAT",
    "INVERSE_VERSION",
    "LoadedCase",
    "load_case",
    "snapshot_case",
    "validate_case",
]
