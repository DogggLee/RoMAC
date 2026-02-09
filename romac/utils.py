import os
from pathlib import Path
from typing import Dict

import yaml


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_run_dir(base_dir: str, run_name: str) -> Path:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    run_dir = Path(base_dir) / f"{run_name}-{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir
