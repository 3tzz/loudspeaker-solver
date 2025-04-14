import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks


def perform_fft(signal, sampling_rate):
    """Wykonuje FFT i zwraca częstotliwości i amplitudy"""
    N = len(signal)
    freqs = fftfreq(N, 1 / sampling_rate)
    fft_values = fft(signal)
    return freqs[: N // 2], 2.0 / N * np.abs(fft_values[: N // 2])


def detect_peaks(freqs, amplitudes, height=None, threshold=None):
    """Wykrywa piki w FFT"""
    peaks, _ = find_peaks(amplitudes, height=height, threshold=threshold)
    return freqs[peaks], amplitudes[peaks]


def forced_oscillator(t, y, freqs, amplitudes, m, c, k):
    """Równanie ruchu z wymuszeniem wieloczęstotliwościowym"""
    x, v = y
    F_t = sum(A * np.cos(2 * np.pi * f * t) for A, f in zip(amplitudes, freqs))
    return [v, (F_t - c * v - k * x) / m]


def simulate_system(freqs, amplitudes, m=0.01, k=100, c=0.5, T=2.0, fs=500):
    """Symuluje ruch układu"""
    t_eval = np.linspace(0, T, int(T * fs), endpoint=False)
    sol = solve_ivp(
        forced_oscillator,
        (0, T),
        [0, 0],
        t_eval=t_eval,
        args=(freqs, amplitudes, m, c, k),
    )
    return t_eval, sol.y[0]


def plot_results(
    t,
    F_t,
    x,
    freqs_F,
    amplitudes_F,
    freqs_X,
    amplitudes_X,
    peaks_F,
    peaks_A_F,
    peaks_X,
    peaks_A_X,
):
    """Tworzy wykresy dla siły, odpowiedzi układu i FFT"""
    fig, axs = plt.subplots(4, 1, figsize=(10, 10))

    axs[0].plot(t, F_t, label="Siła wymuszająca F(t)")
    axs[0].set_ylabel("Siła [N]")
    axs[0].set_title("Siła wymuszająca w czasie")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t, x, label="Przemieszczenie x(t)", color="r")
    axs[1].set_ylabel("Wychylenie [m]")
    axs[1].set_title("Odpowiedź układu drgającego")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].stem(freqs_F, amplitudes_F, basefmt=" ", label="FFT Siły")
    axs[2].plot(peaks_F, peaks_A_F, "ro", label="Piki")
    axs[2].set_xlabel("Częstotliwość [Hz]")
    axs[2].set_ylabel("Amplituda")
    axs[2].set_title("FFT Siły wymuszającej")
    axs[2].grid()
    axs[2].legend()

    axs[3].stem(freqs_X, amplitudes_X, basefmt=" ", label="FFT Przemieszczenia")
    axs[3].plot(peaks_X, peaks_A_X, "ro", label="Piki")
    axs[3].set_xlabel("Częstotliwość [Hz]")
    axs[3].set_ylabel("Amplituda")
    axs[3].set_title("FFT Przemieszczenia x(t)")
    axs[3].grid()
    axs[3].legend()

    plt.tight_layout()
    plt.show()


def main():
    T, fs = 2.0, 500  # Czas symulacji, częstotliwość próbkowania
    t = np.linspace(0, T, int(T * fs), endpoint=False)
    # NOTE: to elaborate
    threshold_F = 0.1
    threshold_x = 0.001

    # F_t = 1.0 * np.cos(2 * np.pi * 10 * t) + 0.5 * np.cos(
    #     2 * np.pi * 25 * t
    # )  # Siła wymuszająca

    F_t = (
        1.0 * np.cos(2 * np.pi * 10 * t)
        + 10.0 * np.cos(2 * np.pi * 100 * t)
        + 55.0 * np.cos(2 * np.pi * 440 * t)
    )

    freqs_F, amplitudes_F = perform_fft(F_t, fs)
    peaks_F, peaks_A_F = detect_peaks(freqs_F, amplitudes_F, threshold=threshold_F)
    print(peaks_F)

    t, x = simulate_system(peaks_F, peaks_A_F)
    freqs_X, amplitudes_X = perform_fft(x, fs)
    peaks_X, peaks_A_X = detect_peaks(freqs_X, amplitudes_X, threshold=threshold_x)
    print(peaks_X)

    plot_results(
        t,
        F_t,
        x,
        freqs_F,
        amplitudes_F,
        freqs_X,
        amplitudes_X,
        peaks_F,
        peaks_A_F,
        peaks_X,
        peaks_A_X,
    )


if __name__ == "__main__":
    main()
