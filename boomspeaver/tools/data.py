import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from dotenv import load_dotenv


def load_json_file(input_path: Path) -> dict[str, float]:
    """Load json file."""
    assert isinstance(input_path, Path)
    assert input_path.exists()
    assert input_path.suffix == ".json"
    with open(input_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    return loaded_data


def save_json_file(output_path: Path, data: dict | list) -> None:
    """Save json file."""
    assert isinstance(output_path, Path)
    assert output_path.suffix == ".json"
    assert isinstance(data, dict) or isinstance(data, list)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Data is not JSON serializable: {e}") from e

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json_data)
    except Exception as e:
        raise IOError(f"Failed to save JSON file: {e}") from e


def load_numpy_file(input_path: Path) -> np.ndarray:
    """Load numpy array from file."""
    assert isinstance(input_path, Path)
    assert input_path.exists()
    assert input_path.suffix == ".npy"
    return np.load(input_path)


def save_numpy_file(output_path: Path, data: np.ndarray) -> None:
    assert isinstance(output_path, Path)
    assert output_path.suffix == ".npy"
    assert isinstance(data, np.ndarray)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.save(output_path, data)
    except Exception as e:
        raise IOError(f"Failed to save numpy file: {e}") from e


def load_wave_file(file_path: Path) -> tuple[np.ndarray, int]:
    """Load a mono waveform from a WAV file using soundfile."""
    assert isinstance(file_path, Path)
    assert file_path.is_file()
    data, fs = sf.read(str(file_path), dtype="float32")
    assert data.ndim == 1
    return data, fs


def save_wave_file(file_path: Path, fs: int, data: np.ndarray) -> None:
    """Save a mono waveform to a WAV file using soundfile."""
    assert isinstance(file_path, Path)
    assert isinstance(fs, int) and fs > 0
    assert isinstance(data, np.ndarray)
    assert data.ndim == 1
    assert np.issubdtype(data.dtype, np.floating) or np.issubdtype(
        data.dtype, np.integer
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(file=str(file_path), data=data, samplerate=fs)


def pad_vector(vector_to_pad: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """Pad vector according to reference vector length."""
    pad_length = len(reference_vector) - len(vector_to_pad)

    if pad_length > 0:
        return np.pad(vector_to_pad, (0, pad_length), mode="constant")
    elif pad_length == 0:
        return vector_to_pad
    else:
        raise ValueError("Provided vector to pad is longer than reference.")


def get_value_from_dict(data: dict, *keys, default=None) -> Any:
    """Safely get a value from nested dictionaries with a default fallback."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def get_env_variables(env_file: Path | None = None) -> None:
    """Load repository bash variables."""
    assert isinstance(env_file, Path) or env_file is None
    if env_file is None:
        env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
    else:
        raise FileNotFoundError(f".env file not found at {env_file}")


def get_repo_dir(run_type: str = "docker") -> Path:
    """Get root repository directory."""
    if run_type == "docker":
        if "WORKING_DIR" not in os.environ:
            get_env_variables()
        if "WORKING_DIR" not in os.environ:
            raise EnvironmentError(
                "WORKING_DIR not found in environment variables even after loading .env"
            )
        repo_dir = Path(os.environ["WORKING_DIR"])
    elif run_type == "python":
        repo_dir = Path(__file__).resolve().parent.parent.parent
    else:
        raise ValueError(f"Unknown run type: {run_type}.")
    if not repo_dir.exists():
        raise FileNotFoundError("Root repository directory not found.")
    return repo_dir
