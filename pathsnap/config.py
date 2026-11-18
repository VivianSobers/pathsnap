"""Load the shared YAML config and resolve its paths against the repository root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the config as a dict, with every entry under ``paths`` turned into an absolute Path."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {name: (REPO_ROOT / p).resolve() for name, p in cfg.get("paths", {}).items()}
    return cfg
