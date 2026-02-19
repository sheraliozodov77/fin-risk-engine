"""Load YAML configs with project root resolution."""
from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(cfg: dict, base: Path) -> None:
    """Resolve relative paths under data and outputs."""
    for key in ("data", "outputs"):
        if key not in cfg or not isinstance(cfg[key], dict):
            continue
        for k, v in cfg[key].items():
            if isinstance(v, str) and not Path(v).is_absolute():
                cfg[key][k] = str(base / v)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, str) and not Path(v2).is_absolute():
                        cfg[key][k][k2] = str(base / v2)


def get_paths(base_dir: Path | None = None) -> dict:
    base = base_dir or _project_root()
    path_file = base / "config" / "paths.yaml"
    if not path_file.exists():
        return {}
    with open(path_file) as f:
        cfg = yaml.safe_load(f) or {}
    _resolve(cfg, base)
    return cfg


def get_model_config(base_dir: Path | None = None) -> dict:
    base = base_dir or _project_root()
    path_file = base / "config" / "model_config.yaml"
    if not path_file.exists():
        return {}
    with open(path_file) as f:
        return yaml.safe_load(f) or {}
