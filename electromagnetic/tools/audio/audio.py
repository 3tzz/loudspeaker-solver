import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

from tools.plot.configs import Axis, Line, Plotter, Points, Subplot
from tools.plot.multi_plotter import MultiPlotter


def fft_analysis(signal, sampling_rate):
    """Perform FFT and return frequencies and amplitudes."""
    N = len(signal)
    freqs = fftfreq(N, 1 / sampling_rate)
    fft_values = fft(signal)
    return freqs[: N // 2], 2.0 / N * np.abs(fft_values[: N // 2])


def get_frequency_resolution(sampling_rate, num_samples):
    """
    Calculate the frequency resolution of a signal.
    """
    return sampling_rate / num_samples


def detect_peaks(signal, height=None, threshold=None):
    """Detects Peaks in signal"""
    peaks, _ = find_peaks(signal, height=height, threshold=threshold)
    return peaks, signal[peaks]


def low_pass_filter(signal_data, fs, cutoff_freq):
    """
    Filters the signal to preserve frequencies below
    `cutoff_freq`and attenuate higher frequencies.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    b, a = signal.butter(4, normal_cutoff, btype="low", analog=False)

    filtered_signal = signal.filtfilt(b, a, signal_data)

    return filtered_signal


def normalize(signal):
    """Normalizes the audio signal to the range [-1, 1]."""
    max_amplitude = np.max(np.abs(signal))
    if max_amplitude > 0:
        return signal / max_amplitude
    return signal


def high_pass_filter(signal, fs, cutoff_freq):
    """
    Apply a high-pass filter to the signal to
    remove frequencies below the cutoff frequency.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(
        4, normal_cutoff, btype="high", analog=False
    )  # 4th order Butterworth filter

    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal


def main():
    from tools.audio.make_sound import generate_chord  # to prevent circular import

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

    print(filtered_freqs)
    print(filtered_signal)

    # Plot the results
    plotter_config = Plotter(
        figsize=(10, 10), grid=True, tight_layout=True, sharex=False, sharey=False
    )
    plotter = MultiPlotter(config=plotter_config)
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Orginal Signal",
                xlabel="Time [s]",
                ylabel="Amplitude",
            ),
            chart_elements=[Line(t, signal, label="signal")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Orginal FFT",
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
