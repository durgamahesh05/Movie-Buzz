from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"
ROOT_ENV_FILE = BASE_DIR.parent / ".env"


def load_env_file(env_file: Path = ENV_FILE) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value

# Load backend-local settings first, then fill any missing values from the repo root.
load_env_file(ENV_FILE)
load_env_file(ROOT_ENV_FILE)


def env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    return default


def env_int(*names: str, default: int) -> int:
    for name in names:
        raw_value = os.getenv(name)
        if raw_value is None or raw_value.strip() == "":
            continue
        raw_value = raw_value.split("#", 1)[0].strip()
        try:
            return int(raw_value)
        except ValueError:
            return default
    return default


def env_path(*names: str, default: str | Path) -> Path:
    return Path(env(*names, default=str(default))).expanduser()
