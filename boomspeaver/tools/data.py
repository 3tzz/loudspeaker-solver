import json
import os
from pathlib import Path
from typing import Any


def load_json_file(input_path: Path) -> dict[str, float]:
    """Load json file."""
    assert isinstance(input_path, Path)
    assert input_path.exists()
    assert input_path.suffix == ".json"
    with open(input_path, "r") as f:
        loaded_data = json.load(f)
    return loaded_data


def get_value_from_dict(data: dict, *keys, default=None) -> Any:
    """Safely get a value from nested dictionaries with a default fallback."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def get_repo_dir() -> Path:
    """Get root repository directory."""
    return Path(os.environ["WORKING_DIR"])
