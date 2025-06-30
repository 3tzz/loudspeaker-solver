import argparse
from pathlib import Path

import numpy as np

from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.data import load_numpy_file, save_numpy_file


def calculate_maxwell_force(
    current_signal: np.ndarray, coil_length: float, magnetic_field: float
) -> np.ndarray:
    """Calculates the Maxwell force on the voice coil based on the current, coil length, and magnetic field."""
    force = current_signal * coil_length * magnetic_field
    return force


def calculate_maxwell_force_from_force_factor(
    current_signal: np.ndarray, magnetic_force_factor: float
) -> np.ndarray:
    """Calculates the Maxwell force on the voice coil based on the current and magnetic force factor."""
    force = current_signal * magnetic_force_factor
    return force


def magnetic_force(
    input_signal_path: Path, loudspeaker_params: Loudspeaker, output_path: Path
) -> None:
    assert isinstance(input_signal_path, Path)
    assert input_signal_path.exists()
    assert input_signal_path.suffix == ".npy"
    assert isinstance(output_path, Path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert isinstance(loudspeaker_params, Loudspeaker)

    magnetic_force_factor = loudspeaker_params.magnet.Bl

    signal = load_numpy_file(input_signal_path)

    force_signal = calculate_maxwell_force_from_force_factor(
        signal, magnetic_force_factor
    )

    save_numpy_file(output_path=output_path, data=force_signal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal current-magnetic_force converter for loudspeaker signals."
    )
    parser.add_argument(
        "--input_signal_path",
        type=str,
        default="examples/log_sweep_nominal_impedance.npy",
        help="Input current signal.",
    )
    parser.add_argument(
        "--loudspeaker_params",
        type=str,
        default="example/prv_audio_6MB400_8ohm.json",
        help="File representing loudspeaker parameters.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/log_sweep_magnetic_force.npy",
        help="Output path.",
    )

    args = parser.parse_args()

    magnetic_force(
        input_signal_path=Path(args.input_signal_path),
        loudspeaker_params=Loudspeaker.from_json(Path(args.loudspeaker_params)),
        output_path=Path(args.output_path),
    )
