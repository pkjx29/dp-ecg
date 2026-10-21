"""Load the YAML configs in configs/. One source of truth per setting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
ARTIFACTS = PROJECT_ROOT / "artifacts"


def load_config(name: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    path = (config_dir or CONFIG_DIR) / f"{name}.yaml"
    with path.open() as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return cfg


def resolve(path: str | Path) -> Path:
    """Relative paths in configs are relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def artifact_dir(kind: str) -> Path:
    """artifacts/<kind>/, created on demand (metrics, figures, reports, checkpoints, logs)."""
    d = ARTIFACTS / kind
    d.mkdir(parents=True, exist_ok=True)
    return d
