import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
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


def load_csv_file(input_path: Path, header: bool = False) -> pd.DataFrame:
    """Load CSV file into pandas DataFrame."""
    assert isinstance(input_path, Path)
    assert input_path.exists()
    assert input_path.suffix == ".csv"

    try:
        if header:
            df = pd.read_csv(input_path)
        else:
            df = pd.read_csv(input_path, header=None)
    except Exception as e:
        raise IOError(f"Failed to load CSV file: {e}") from e

    return df


def save_csv_file(output_path: Path, df: pd.DataFrame) -> None:
    """Save pandas DataFrame to CSV file."""
    assert isinstance(output_path, Path)
    assert output_path.suffix == ".csv"
    assert isinstance(df, pd.DataFrame)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise IOError(f"Failed to save CSV file: {e}") from e


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


def save_txt_file(data: list[str], file_path: Path) -> None:
    """
    Save a list of strings to a .txt file, one item per line.
    """
    assert isinstance(file_path, Path)
    assert isinstance(data, list)
    assert len(data) != 0
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt_file(file_path: Path) -> list[str]:
    """
    Load a list of strings from a .txt file, assuming one item per line.
    """
    assert file_path.exists()

    with file_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pad_vector(
    array_to_pad: np.ndarray, reference_array: np.ndarray, dim: int = 0
) -> np.ndarray:
    """
    Pad a 1D or 2D array along the specified dimension to match the reference array.
    """
    if array_to_pad.ndim not in (1, 2) or reference_array.ndim not in (1, 2):
        raise ValueError("Only 1D and 2D arrays are supported.")
    if dim >= array_to_pad.ndim:
        raise ValueError(
            f"Cannot pad dimension {dim} of an array with shape {array_to_pad.shape}"
        )

    pad_len = reference_array.shape[dim] - array_to_pad.shape[dim]
    if pad_len < 0:
        raise ValueError(
            "Array to pad is longer than reference along the specified dimension."
        )
    if pad_len == 0:
        return array_to_pad

    if array_to_pad.ndim == 1:
        pad_width = (0, pad_len) if dim == 0 else (pad_len, 0)
        return np.pad(array_to_pad, pad_width, mode="constant")
    else:
        pad_config = [(0, 0), (0, 0)]
        pad_config[dim] = (0, pad_len)
        return np.pad(array_to_pad, pad_config, mode="constant")


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


def search4paths(
    input_path: Path, search_pattern: str, recursive: bool
) -> Iterable[Path]:
    """Search for file or directory paths matching a pattern."""
    assert isinstance(input_path, Path) and input_path.exists()
    assert isinstance(search_pattern, str)
    assert isinstance(recursive, bool)
    if recursive:
        output_paths = input_path.rglob(search_pattern)
    else:
        output_paths = input_path.glob(search_pattern)
    return output_paths


def remove_paths(paths: Iterable[Path] | list[Path], remove_dirs: bool = False) -> None:
    """Remove files, symlinks, and optionally directories from a given iterator of paths."""
    for path in paths:
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif remove_dirs and path.is_dir():
                shutil.rmtree(path)
            print(f"Removed: {path}")
        except Exception as e:
            print(f"Failed to remove {path}: {e}")


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
        raise FileNotFoundError(f"Root repository directory not found {repo_dir}.")
    return repo_dir
