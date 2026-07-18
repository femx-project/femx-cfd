"""Configuration conversion and snapshot helpers."""

from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf

from femx_cfd._typing import Config, PathLike

from .artifacts import _atomic_path


def to_dict(cfg: DictConfig) -> Config:
    """Resolve a Hydra config and detach numerical code from OmegaConf."""
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("config root must be a mapping")
    return cfg_dict


def save_config(cfg: DictConfig, path: PathLike) -> None:
    """Atomically save the resolved application config as YAML."""
    resolved = OmegaConf.create(to_dict(cfg))
    with _atomic_path(path) as tmp:
        OmegaConf.save(resolved, tmp)


def analysis_output(case: Config) -> Path:
    """Return a stable run path, or Hydra's numbered path for a sweep."""
    hydra = HydraConfig.get()
    if hydra.mode is RunMode.MULTIRUN:
        return Path(hydra.runtime.output_dir)
    return Path(case["prepared"]).parent / case["job"]
