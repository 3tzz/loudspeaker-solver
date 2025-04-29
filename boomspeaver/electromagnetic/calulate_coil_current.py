import argparse
from pathlib import Path

import numpy as np

from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.data import load_wave_file, save_numpy_file
from boomspeaver.tools.signal.signal import process_signal_in_chunks


def voltage_to_current_resistive(voltage_signal: np.ndarray, Re: float) -> np.ndarray:
    """Simple resistive model: i(t) = v(t) / Re"""
    return voltage_signal / Re


def voltage_to_current_nominal(
    voltage_signal: np.ndarray, Z_nominal: float
) -> np.ndarray:
    """Nominal impedance model: i(t) = v(t) / Z_nominal"""
    return voltage_signal / Z_nominal


def voltage_to_current_inductive(
    voltage_signal: np.ndarray, re: float, le: float, fs: int
) -> np.ndarray:
    """Applies a simple R + jωL impedance model to convert voltage to current."""
    frequencies = np.fft.fftfreq(len(voltage_signal), d=1 / fs)
    omega = 2 * np.pi * frequencies
    impedance = re + 1j * omega * le

    impedance[np.abs(impedance) < 1e-12] = 1e-12

    voltage_spectrum = np.fft.fft(voltage_signal)
    current_spectrum = voltage_spectrum / impedance
    current_signal = np.fft.ifft(current_spectrum)
    return np.real(current_signal)


def validate(
    input_signal_path: Path, loudspeaker_params: Loudspeaker, output_path: Path
) -> None:
    assert isinstance(input_signal_path, Path)
    assert input_signal_path.exists()
    assert input_signal_path.suffix == ".wav"
    assert isinstance(output_path, Path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assert isinstance(loudspeaker_params, Loudspeaker)


def inductive(
    input_signal_path: Path,
    loudspeaker_params: Loudspeaker,
    output_path: Path,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> None:
    validate(
        input_signal_path=input_signal_path,
        loudspeaker_params=loudspeaker_params,
        output_path=output_path,
    )

    re = loudspeaker_params.voice_coil.RE
    le = loudspeaker_params.voice_coil.LE

    signal, sampling_rate = load_wave_file(input_signal_path)
    if len(signal.shape) > 1:
        raise ValueError("Only mono audio signals are supported.")

    if chunk_size is None and overlap is None:
        current_signal = voltage_to_current_inductive(signal, re, le, sampling_rate)
    elif all([chunk_size, overlap]):
        current_signal = process_signal_in_chunks(
            signal,
            chunk_size,
            overlap,
            voltage_to_current_inductive,
            re,
            le,
            sampling_rate,
        )
    else:
        raise ValueError("U need to provide both chunk size and overlap(it can be 0).")

    save_numpy_file(output_path=output_path, data=current_signal)


def nominal_impedance(
    input_signal_path: Path, loudspeaker_params: Loudspeaker, output_path: Path
) -> None:
    validate(
        input_signal_path=input_signal_path,
        loudspeaker_params=loudspeaker_params,
        output_path=output_path,
    )

    z_nominal = loudspeaker_params.voice_coil.Z

    signal, _ = load_wave_file(input_signal_path)
    if len(signal.shape) > 1:
        raise ValueError("Only mono audio signals are supported.")

    current_signal = voltage_to_current_nominal(signal, z_nominal)

    save_numpy_file(output_path=output_path, data=current_signal)


def resistive(
    input_signal_path: Path, loudspeaker_params: Loudspeaker, output_path: Path
) -> None:
    validate(
        input_signal_path=input_signal_path,
        loudspeaker_params=loudspeaker_params,
        output_path=output_path,
    )

    resistance = loudspeaker_params.voice_coil.RE

    signal, _ = load_wave_file(input_signal_path)
    if len(signal.shape) > 1:
        raise ValueError("Only mono audio signals are supported.")

    current_signal = voltage_to_current_resistive(signal, resistance)

    # Save result
    save_numpy_file(output_path=output_path, data=current_signal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal voltage-to-current converter for loudspeaker signals."
    )
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "--input_signal_path",
        type=str,
        default="examples/log_sweep.wav",
        help="input audio signal.",
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
        default="output/log_sweep_current.npy",
        help="Output path.",
    )

    resistive_parser = subparsers.add_parser("resistive", help="The simplest version.")
    nominal_parser = subparsers.add_parser("nominal", help="Average impedance.")
    inductive_parser = subparsers.add_parser(
        "inductive", help="Frequency dependend via inductance."
    )

    inductive_parser.add_argument(
        "--chunk",
        action="store_true",
        help="Process signal in chunks.",
    )
    inductive_parser.add_argument(
        "--size",
        type=int,
        default=4096,
        help="Chunk size.",
    )
    inductive_parser.add_argument(
        "--overlap",
        type=int,
        default=512,
        help="How many samples chunks should overlap.",
    )

    args = parser.parse_args()

    if args.command == "resistive":
        resistive(
            input_signal_path=Path(args.input_signal_path),
            loudspeaker_params=args.loudspeaker_params,
            output_path=Path(args.output_path),
        )
    elif args.command == "nominal":
        nominal_impedance(
            input_signal_path=Path(args.input_signal_path),
            loudspeaker_params=args.loudspeaker_params,
            output_path=Path(args.output_path),
        )
    elif args.command == "inductive":
        if args.chunk is True:
            inductive(
                input_signal_path=Path(args.input_signal_path),
                loudspeaker_params=args.loudspeaker_params,
                output_path=Path(args.output_path),
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        else:
            inductive(
                input_signal_path=Path(args.input_signal_path),
                loudspeaker_params=args.loudspeaker_params,
                output_path=Path(args.output_path),
            )
    else:
        raise NotImplementedError(f"Unknown processing type: {args.command}.")
