import numpy as np
from scipy.integrate import solve_ivp

from tools.audio.audio import detect_peaks, fft_analysis
from tools.audio.make_sound import generate_chord, normalize
from tools.plot.configs import Axis, Line, Plotter, Points, Stem, Subplot
from tools.plot.multi_plotter import MultiPlotter


def analyse_steady_state_signal(signal, fs, peak_height=None, peak_threshold=None):
    """Analyse steady state signal using fft and peak detection"""
    print(type(signal))
    assert isinstance(signal, (list, np.ndarray)) and isinstance(fs, int)

    frequencies, amplitudes = fft_analysis(signal, fs)
    idxs, signal_filtered = detect_peaks(
        amplitudes, height=peak_height, threshold=peak_threshold
    )
    return frequencies[idxs], signal_filtered


def main():
    # Generate signal
    duration = 2.0  # [s]
    fs = 2000  # [Hz]

    chord_frequencies = [440, 523.25, 659.25]  # A minor chord frequencies (A, C, E)
    amplitudes_weight = [20, 60, 100]

    t, signal = generate_chord(chord_frequencies, duration, fs, amplitudes_weight)

    # Analyse signal
    frequencies_filtered, signal_filtered = analyse_steady_state_signal(signal, fs)

    # Analyse normalized signal
    signal_normalized = normalize(signal)
    frequencies_norm_filtered, signal_norm_filtered = analyse_steady_state_signal(
        signal_normalized, fs
    )

    # Result
    print(f"Original signal frequencies: {chord_frequencies}")
    print(f"Detected frequencies: {frequencies_filtered}")
    print(f"Detected frequencies form normalized signal: {frequencies_norm_filtered}")

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
                Stem(frequencies_filtered, signal_filtered, label="fft"),
            ],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Normalized Signal",
                xlabel="Time [s]",
                ylabel="Amplitude",
            ),
            chart_elements=[Line(t, signal_normalized, label="normalized signal")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Normalized FFT",
                xlabel="Frequency [Hz]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Stem(frequencies_norm_filtered, signal_norm_filtered, label="fft"),
            ],
        )
    )
    plotter.plot()


if __name__ == "__main__":
    main()
