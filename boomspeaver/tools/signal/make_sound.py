import argparse
from pathlib import Path

import numpy as np

from boomspeaver.tools.data import save_wave_file
from boomspeaver.tools.signal.signal import normalize


def generate_time_domain(
    sampling_rate: int, duration: float | None = None, n_samples: int | None = None
) -> np.ndarray:
    """Generate time domain vector."""
    assert isinstance(sampling_rate, int)
    assert isinstance(duration, float) or duration is None
    assert isinstance(n_samples, int) or n_samples is None

    if duration is not None and n_samples is None:
        time = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    elif n_samples is not None and duration is None:
        time = np.linspace(0, n_samples / sampling_rate, n_samples, endpoint=False)
    else:
        raise ValueError(
            f"You need to provide duration [{bool(duration is not None)}] or number of samples [{bool(n_samples is not None)}]."
        )
    return time


def generate_cosine_wave(
    frequency: float,
    duration: float,
    fs: int,
    amplitude: float,
    signal_norm: bool = False,
) -> np.ndarray:
    """Generates a sine wave signal."""
    t = generate_time_domain(sampling_rate=fs, duration=duration)
    cosine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    if signal_norm:
        cosine_wave = normalize(cosine_wave)
    return cosine_wave


def generate_sweep_wave(
    f_start: float,
    f_end: float,
    duration: float,
    fs: int,
    amplitude: float,
    signal_norm: bool = False,
) -> np.ndarray:
    """Generates a linear sweep signal (chirp)."""

    t = generate_time_domain(sampling_rate=fs, duration=duration)
    sweep_signal = amplitude * np.sin(
        2 * np.pi * np.linspace(f_start, f_end, len(t)) * t
    )
    if signal_norm:
        sweep_signal = normalize(sweep_signal)
    return sweep_signal


def generate_log_sweep(
    f_start: float,
    f_end: float,
    duration: int,
    fs: int,
    amplitude: float,
    signal_norm: bool = False,
) -> np.ndarray:
    """Generates a logarithmic frequency sweep (chirp) with constant perceived volume."""

    t = generate_time_domain(sampling_rate=fs, duration=duration)
    log_frequencies = np.logspace(np.log10(f_start), np.log10(f_end), num=len(t))
    sweep_signal = amplitude * np.sin(2 * np.pi * log_frequencies * t)
    if signal_norm:
        sweep_signal = normalize(sweep_signal)
    return sweep_signal


def generate_chord(
    frequencies: list[float],
    duration: float,
    fs: int,
    amplitude: float,
    signal_norm: bool = False,
) -> np.ndarray:
    """Generates a chord by combining multiple sine waves."""

    t = generate_time_domain(sampling_rate=fs, duration=duration)
    signal = np.zeros_like(t)
    for f in frequencies:
        signal += generate_cosine_wave(f, duration, fs, amplitude)
    if signal_norm:
        signal = normalize(signal)
    return signal


def main():
    parser = argparse.ArgumentParser(description="Generate audio signals.")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for cosine wave
    cosine_parser = subparsers.add_parser("cosine", help="Generate a cosine wave.")
    cosine_parser.add_argument(
        "--frequency",
        type=float,
        default=440,
        help="Frequency of the cosine wave in Hz (default 440 Hz)",
    )
    cosine_parser.add_argument(
        "--duration", type=float, default=1, help="Duration in seconds (default 1 s)"
    )
    cosine_parser.add_argument(
        "--fs", type=int, default=48000, help="Sampling frequency (default 48000 Hz)"
    )
    cosine_parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude (0-1, default 0.5)"
    )
    cosine_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/cosine_wave.wav"),
        help="Output file path (default output/cosine_wave.wav)",
    )
    cosine_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable the feature (default: disabled)",
    )

    # Subparser for SWEEP wave (linear sweep)
    sweep_parser = subparsers.add_parser(
        "sweep", help="Generate a linear sweep wave (chirp)."
    )
    sweep_parser.add_argument(
        "--f_start",
        type=float,
        default=20,
        help="Start frequency in Hz (default 20 Hz)",
    )
    sweep_parser.add_argument(
        "--f_end",
        type=float,
        default=20000,
        help="End frequency in Hz (default 20 kHz)",
    )
    sweep_parser.add_argument(
        "--duration", type=float, default=1, help="Duration in seconds (default 1 s)"
    )
    sweep_parser.add_argument(
        "--fs", type=int, default=48000, help="Sampling frequency (default 48000 Hz)"
    )
    sweep_parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude (0-1, default 0.5)"
    )
    sweep_parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/sweep_signal.wav"),
        help="Output file path (default output/sweep_signal.wav)",
    )
    sweep_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable the feature (default: disabled)",
    )

    # Subparser for LOG SWEEP (logarithmic sweep)
    log_sweep_parser = subparsers.add_parser(
        "log_sweep", help="Generate a logarithmic sweep wave (chirp)."
    )
    log_sweep_parser.add_argument(
        "--f_start",
        type=float,
        default=20,
        help="Start frequency in Hz (default 20 Hz)",
    )
    log_sweep_parser.add_argument(
        "--f_end",
        type=float,
        default=20000,
        help="End frequency in Hz (default 20 kHz)",
    )
    log_sweep_parser.add_argument(
        "--duration", type=float, default=1, help="Duration in seconds (default 1 s)"
    )
    log_sweep_parser.add_argument(
        "--fs", type=int, default=48000, help="Sampling frequency (default 48000 Hz)"
    )
    log_sweep_parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude (0-1, default 0.5)"
    )
    log_sweep_parser.add_argument(
        "--output",
        type=Path,
        default=Path("cosine/log_sweep_signal.wav"),
        help="Output file path (default output/log_sweep_signal.wav)",
    )
    log_sweep_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable the feature (default: disabled)",
    )

    # Subparser for CHORD
    chord_parser = subparsers.add_parser(
        "chord", help="Generate a chord by combining sine waves."
    )
    chord_parser.add_argument(
        "--frequencies",
        nargs="+",
        type=float,
        default=[440, 554.37, 659.25],
        help="List of frequencies for the chord (default is A major chord 440 Hz, 554.37 Hz, 659.25 Hz)",
    )
    chord_parser.add_argument(
        "--duration", type=float, default=1, help="Duration in seconds (default 1 s)"
    )
    chord_parser.add_argument(
        "--fs", type=int, default=48000, help="Sampling frequency (default 48000 Hz)"
    )
    chord_parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude (0-1, default 0.5)"
    )
    chord_parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/chord_signal.wav"),
        help="Output file path (default output/chord_signal.wav)",
    )
    chord_parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable the feature (default: disabled)",
    )

    args = parser.parse_args()

    if args.command == "cosine":
        signal_wave = generate_cosine_wave(
            args.frequency, args.duration, args.fs, args.amplitude, args.normalize
        )
        print(signal_wave.shape)
        save_wave_file(
            args.output, args.fs, np.int16(signal_wave * 32767)
        )  # Convert to 16-bit PCM
    elif args.command == "sweep":
        signal_wave = generate_sweep_wave(
            args.f_start,
            args.f_end,
            args.duration,
            args.fs,
            args.amplitude,
            args.normalize,
        )
        save_wave_file(
            args.output, args.fs, np.int16(signal_wave * 32767)
        )  # Convert to 16-bit PCM
    elif args.command == "log_sweep":
        signal_wave = generate_log_sweep(
            args.f_start,
            args.f_end,
            args.duration,
            args.fs,
            args.amplitude,
            args.normalize,
        )
        save_wave_file(
            args.output, args.fs, np.int16(signal_wave * 32767)
        )  # Convert to 16-bit PCM
    elif args.command == "chord":
        signal_wave = generate_chord(
            args.frequencies, args.duration, args.fs, args.amplitude, args.normalize
        )
        save_wave_file(
            args.output, args.fs, np.int16(signal_wave * 32767)
        )  # Convert to 16-bit PCM


if __name__ == "__main__":
    main()
