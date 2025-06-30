import argparse
from pathlib import Path

import numpy as np

from boomspeaver.loudspeaker.electrical_impedance import ImpedanceData
from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.data import load_wave_file, save_numpy_file
from boomspeaver.tools.signal.signal import istft, plot_spectrogram, stft


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
    input_signal_path: Path,
    output_path: Path,
    loudspeaker_params: Loudspeaker | None = None,
) -> None:
    assert isinstance(input_signal_path, Path)
    assert input_signal_path.exists()
    assert input_signal_path.suffix == ".wav"
    assert isinstance(output_path, Path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if loudspeaker_params is not None:
        assert isinstance(loudspeaker_params, Loudspeaker)


def frequency_impedance(
    input_signal_path: Path,
    impedance_params: Path,
    output_path: Path,
    n_fft: int,
    win_length: int,
    hop_length: int,
) -> None:
    """Calculate circuit frequency dependent."""

    validate(input_signal_path, output_path)

    signal, sampling_rate = load_wave_file(input_signal_path)
    if len(signal.shape) > 1:
        raise ValueError("Only mono audio signals are supported.")

    impedance = ImpedanceData.from_csv(impedance_params)
    impedance.extrapolate()  # NOTE: frequency range could be provided here

    freqs, spec = stft(
        signal=signal,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )

    assert len(freqs) == spec.shape[0]
    for idx_freq, values_per_freq in enumerate(spec):
        freq = freqs[idx_freq]
        for idx_time, voltage_in_time in enumerate(values_per_freq):
            z_total = impedance.get_impedance(freq)
            assert z_total != 0
            amperage_value = voltage_in_time / z_total
            spec[idx_freq][idx_time] = amperage_value

    current_signal = istft(
        spec=spec,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        length=len(signal),
    )

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
        help="Input audio signal.",
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
    frequency_impedance_parser = subparsers.add_parser(
        "frequency_impedance",
        help="Analyze frequency-dependent electrical impedance using total impedance data.",
    )

    frequency_impedance_parser.add_argument(
        "--impedance_params",
        type=str,
        default="example/electrical_impedance.csv",
        help="File representing total electrical impedance according to frequencies.",
    )
    frequency_impedance_parser.add_argument(
        "--n_fft",
        type=int,
        default=2048,
        help="Number of FFT points. Defines the frequency resolution of the STFT.",
    )
    frequency_impedance_parser.add_argument(
        "--win_length",
        type=int,
        default=2048,
        help="Length of the window applied to each segment for STFT. Should be <= n_fft.",
    )
    frequency_impedance_parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Number of samples between successive STFT windows.",
    )

    args = parser.parse_args()

    if args.command == "resistive":
        resistive(
            input_signal_path=Path(args.input_signal_path),
            loudspeaker_params=Loudspeaker.from_json(Path(args.loudspeaker_params)),
            output_path=Path(args.output_path),
        )
    elif args.command == "nominal":
        nominal_impedance(
            input_signal_path=Path(args.input_signal_path),
            loudspeaker_params=Loudspeaker.from_json(Path(args.loudspeaker_params)),
            output_path=Path(args.output_path),
        )
    elif args.command == "frequency_impedance":
        frequency_impedance(
            input_signal_path=Path(args.input_signal_path),
            impedance_params=args.impedance_params,
            output_path=Path(args.output_path),
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
        )
    else:
        raise NotImplementedError(f"Unknown processing type: {args.command}.")
