import argparse
import os
import wave
from pathlib import Path

import numpy as np


def generate_sine_wave(frequency, duration, fs, amplitude):
    """Generates a sine wave signal."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return normalize(sine_wave)


def generate_sweep_wave(f_start, f_end, duration, fs, amplitude):
    """Generates a linear sweep signal (chirp)."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sweep_signal = amplitude * np.sin(
        2 * np.pi * np.linspace(f_start, f_end, len(t)) * t
    )
    return normalize(sweep_signal)


def generate_log_sweep(f_start, f_end, duration, fs, amplitude):
    """Generates a logarithmic frequency sweep (chirp) with constant perceived volume."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    log_frequencies = np.logspace(np.log10(f_start), np.log10(f_end), num=len(t))
    sweep_signal = amplitude * np.sin(2 * np.pi * log_frequencies * t)
    return normalize(sweep_signal)


def generate_chord(frequencies, duration, fs, amplitude):
    """Generates a chord by combining multiple sine waves."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    signal = np.zeros_like(t)
    for f in frequencies:
        signal += generate_sine_wave(f, duration, fs, amplitude)
    return normalize(signal)


def normalize(signal):
    """Normalizes the audio signal to the range [-1, 1]."""
    max_amplitude = np.max(np.abs(signal))
    if max_amplitude > 0:
        return signal / max_amplitude
    return signal


def save_wave_file(file_path, fs, data):
    """Saves the waveform to a WAV file."""
    with wave.open(str(file_path), "w") as wav_file:  # Convert Path to string
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(fs)
        wav_file.writeframes(data.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Generate audio signals.")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for SINE wave
    sine_parser = subparsers.add_parser("sine", help="Generate a sine wave.")
    sine_parser.add_argument(
        "--frequency",
        type=float,
        default=440,
        help="Frequency of the sine wave in Hz (default 440 Hz)",
    )
    sine_parser.add_argument(
        "--duration", type=float, default=1, help="Duration in seconds (default 1 s)"
    )
    sine_parser.add_argument(
        "--fs", type=int, default=44100, help="Sampling frequency (default 44100 Hz)"
    )
    sine_parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude (0-1, default 0.5)"
    )
    sine_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sine_wave.wav"),
        help="Output file path (default output/sine_wave.wav)",
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
        "--fs", type=int, default=44100, help="Sampling frequency (default 44100 Hz)"
    )
    sweep_parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude (0-1, default 0.5)"
    )
    sweep_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/sweep_signal.wav"),
        help="Output file path (default output/sweep_signal.wav)",
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
        "--fs", type=int, default=44100, help="Sampling frequency (default 44100 Hz)"
    )
    log_sweep_parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude (0-1, default 0.5)"
    )
    log_sweep_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/log_sweep_signal.wav"),
        help="Output file path (default output/log_sweep_signal.wav)",
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
        "--fs", type=int, default=44100, help="Sampling frequency (default 44100 Hz)"
    )
    chord_parser.add_argument(
        "--amplitude", type=float, default=0.5, help="Amplitude (0-1, default 0.5)"
    )
    chord_parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/chord_signal.wav"),
        help="Output file path (default output/chord_signal.wav)",
    )

    args = parser.parse_args()

    if args.command == "sine":
        signal_wave = generate_sine_wave(
            args.frequency, args.duration, args.fs, args.amplitude
        )
        save_wave_file(
            args.output, args.fs, np.int16(signal_wave * 32767)
        )  # Convert to 16-bit PCM
    elif args.command == "sweep":
        signal_wave = generate_sweep_wave(
            args.f_start, args.f_end, args.duration, args.fs, args.amplitude
        )
        save_wave_file(
            args.output, args.fs, np.int16(signal_wave * 32767)
        )  # Convert to 16-bit PCM
    elif args.command == "log_sweep":
        signal_wave = generate_log_sweep(
            args.f_start, args.f_end, args.duration, args.fs, args.amplitude
        )
        save_wave_file(
            args.output, args.fs, np.int16(signal_wave * 32767)
        )  # Convert to 16-bit PCM
    elif args.command == "chord":
        signal_wave = generate_chord(
            args.frequencies, args.duration, args.fs, args.amplitude
        )
        save_wave_file(
            args.output, args.fs, np.int16(signal_wave * 32767)
        )  # Convert to 16-bit PCM


if __name__ == "__main__":
    main()
