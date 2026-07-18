"""Shared type aliases for femx-cfd's dynamic Python and femx interfaces."""

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Union


Config = Dict[str, Any]
Dynamic = Any
PathLike = Union[str, Path]
ProgressEvent = Mapping[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]
OptionalProgress = Optional[ProgressCallback]
ReportCallback = Callable[[str], None]


class Progress(Protocol):
    """Common interface implemented by console, JSON, and null progress sinks."""

    def __enter__(self) -> "Progress": ...

    def __exit__(self, *args: object) -> None: ...

    def stage(self, message: str) -> None: ...

    def __call__(self, event: ProgressEvent) -> None: ...

    def close(self) -> None: ...
