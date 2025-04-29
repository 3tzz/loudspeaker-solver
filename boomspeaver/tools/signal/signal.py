from typing import Any, Callable

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks, get_window, spectrogram

from boomspeaver.tools.plot.configs import Axis, Line, Plotter, Points, Subplot
from boomspeaver.tools.plot.multi_plotter import MultiPlotter


def validate_signal(signal: np.ndarray) -> None:
    assert isinstance(signal, np.ndarray)
    assert signal.ndim == 1
    assert signal.size >= 2


def fft_analysis(signal: np.ndarray, sampling_rate: int) -> tuple[np.ndarray, Any]:
    """Perform FFT and return frequencies and amplitudes."""
    assert isinstance(sampling_rate, int)
    validate_signal(signal)
    n = len(signal)
    freqs = fftfreq(n, 1 / sampling_rate)
    fft_values = fft(signal)
    return freqs[: n // 2], 2.0 / n * np.abs(fft_values[: n // 2])


def get_resolution(vector: np.ndarray) -> float:
    """Get resolution behind values from vector."""
    validate_signal(vector)
    return float(np.mean(np.diff(vector)))


def detect_peaks(signal: np.ndarray, height: float = None, threshold: float = None):
    """Detects Peaks in signal"""
    assert isinstance(height, float) or height is None
    assert isinstance(threshold, float) or threshold is None
    validate_signal(signal)
    peaks, _ = find_peaks(signal, height=height, threshold=threshold)
    return peaks, signal[peaks]


def detect_signal_bounds(
    signal: np.ndarray, threshold: float | None = None
) -> tuple[int, int] | None:
    """
    Detect start and end sample indices where the signal exceeds a threshold.
    If no threshold is given, it is estimated from the signal.
    """
    validate_signal(signal)
    abs_signal = np.abs(signal)

    if threshold is None:
        threshold = 0.05 * np.max(abs_signal)  # 5% of max amplitude
        if threshold == 0:
            return None  # signal is flat

    above_threshold = abs_signal > threshold

    if not np.any(above_threshold):
        return None  # no signal found

    start_idx = np.argmax(above_threshold)
    end_idx = len(signal) - np.argmax(above_threshold[::-1]) - 1
    return start_idx, end_idx


def process_signal_in_chunks(
    signal: np.ndarray,
    chunk_size: int,
    overlap: int,
    transform_function: Callable,
    *func_args
) -> np.ndarray:
    """Processes the signal with a given impedance model and returns the current signal."""
    current_signal = np.zeros_like(signal)

    num_chunks = (len(signal) - overlap) // (chunk_size - overlap)

    for i in range(num_chunks):
        start_idx = i * (chunk_size - overlap)
        end_idx = start_idx + chunk_size
        chunk = signal[start_idx:end_idx]

        window = get_window("hann", len(chunk))
        chunk_windowed = chunk * window

        current_chunk = transform_function(chunk_windowed, *func_args)

        current_signal[start_idx:end_idx] = current_chunk

    return current_signal


def chunk_signal(signal: np.ndarray, chunk_size: int, overlap: int):
    """Generator that yields chunks of the signal with optional overlap."""
    num_chunks = (len(signal) - overlap) // (chunk_size - overlap)

    for i in range(num_chunks):
        start_idx = i * (chunk_size - overlap)
        end_idx = start_idx + chunk_size
        chunk = signal[start_idx:end_idx]

        window = hann(len(chunk))
        chunk_windowed = chunk * window

        yield chunk_windowed


def normalize(signal: np.ndarray) -> np.ndarray:
    """Normalizes the audio signal to the range [-1, 1]."""
    validate_signal(signal)
    max_amplitude = np.max(np.abs(signal))
    if max_amplitude > 0:
        return signal / max_amplitude
    return signal


def low_pass_filter(signal: np.ndarray, fs: int, cutoff_freq: float):
    """
    Filters the signal to preserve frequencies below
    `cutoff_freq`and attenuate higher frequencies.
    """
    assert isinstance(fs, int)
    assert isinstance(cutoff_freq, float)
    validate_signal(signal)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    b, a = butter(4, normal_cutoff, btype="low", analog=False)

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def high_pass_filter(signal: np.ndarray, fs: int, cutoff_freq: float):
    """
    Apply a high-pass filter to the signal to
    remove frequencies below the cutoff frequency.
    """
    assert isinstance(fs, int)
    assert isinstance(cutoff_freq, float)
    validate_signal(signal)
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype="high", analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal


def calculate_spectrogram(
    signal: np.ndarray, sampling_rate: int
) -> tuple[Any, Any, Any]:
    """Plot a spectrogram of the given signal."""
    validate_signal(signal)

    f, t, Sxx = spectrogram(signal, fs=sampling_rate)
    decibelised = 10 * np.log10(Sxx + 1e-10)
    return t, f, decibelised


def main():
    from boomspeaver.tools.signal.make_sound import (
        generate_chord,
    )  # NOTE:to prevent circular import

    # Define chord frequencies for A minor chord (A, C, E)
    # Frequencies of the A minor chord notes: 440 Hz (A), 523.25 Hz (C), 659.25 Hz (E)
    sampling_rate = 10000  # Sampling rate in Hz
    duration = 2  # Duration of the signal in seconds
    chord_frequencies = [440, 523.25, 659.25]  # A minor chord frequencies (A, C, E)

    # Generate the chord signal
    t, signal = generate_chord(
        chord_frequencies,
        duration,
        sampling_rate,
    )

    # Perform FFT analysis
    freqs, amplitudes = fft_analysis(signal, sampling_rate)

    idxes, filtered_signal = detect_peaks(amplitudes)
    filtered_freqs = freqs[idxes]

    # Plot the results
    plotter_config = Plotter(
        figsize=(10, 10), grid=True, tight_layout=True, sharex=False, sharey=False
    )
    plotter = MultiPlotter(config=plotter_config)
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Original Signal",
                xlabel="Time [s]",
                ylabel="Amplitude",
            ),
            chart_elements=[Line(t, signal, label="signal")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Original FFT",
                xlabel="Frequency [Hz]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Points(
                    filtered_freqs,
                    filtered_signal,
                    label="Detected Peaks",
                    marker="o",
                    color="r",
                ),
                Line(freqs, amplitudes, label="fft"),
            ],
        )
    )
    plotter.plot()


if __name__ == "__main__":
    main()
