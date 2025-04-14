import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

# Parameters for voice coil (DC resistance and inductance are used for impedance calculation)
dc_resistance = 6.1  # DC resistance (Ohms)
inductance = 0.53e-3  # Inductance at 1kHz (H)
sampling_rate = 44100  # Hz (assuming standard WAV file rate)

# Example frequency-dependent impedance (You would replace this with your actual data)
impedance_data = {
    20: 8,  # Example impedance at 20 Hz
    100: 8.1,  # Example impedance at 100 Hz
    1000: 8,  # Example impedance at 1kHz
    2000: 8.3,  # Example impedance at 2kHz
    5000: 8.5,  # Example impedance at 5kHz
    10000: 9,  # Example impedance at 10kHz
}


# Interpolation function to get impedance at any frequency
def get_impedance_at_frequency(frequency, impedance_data):
    frequencies = np.array(list(impedance_data.keys()))
    impedances = np.array(list(impedance_data.values()))
    impedance = np.interp(frequency, frequencies, impedances)
    return impedance


# Load the .wav file
def load_wav(file_path):
    sample_rate, audio_data = wav.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)  # Convert to mono if stereo
    return sample_rate, audio_data


def calculate_current(audio_data, impedance_data, sampling_rate):
    # Compute FFT of the audio data
    fft_values = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(audio_data), 1 / sampling_rate)

    # Only keep positive frequencies
    positive_freqs = freqs[: len(freqs) // 2]
    positive_fft_values = fft_values[: len(fft_values) // 2]

    # Initialize current signal in time domain as float64
    current_signal = np.zeros_like(audio_data, dtype=np.float64)  # Change to float64

    # Loop over positive frequencies to compute the current at each frequency
    for i, f in enumerate(positive_freqs):
        impedance = get_impedance_at_frequency(f, impedance_data)
        V_f = positive_fft_values[i]
        I_f = V_f / impedance
        current_signal += np.real(I_f) * np.cos(
            2
            * np.pi
            * f
            * np.linspace(0, len(audio_data) / sampling_rate, len(audio_data))
        )  # Rebuild current in time domain

    return positive_freqs, current_signal


# Plot the results (Magnitude of Voltage and Current vs Frequency)
def plot_results(freqs, audio_data, current_signal):
    plt.figure(figsize=(12, 6))

    # Plot the magnitude of the FFT of audio_data (Voltage)
    fft_audio_data = np.fft.fft(audio_data)
    positive_freqs = freqs
    positive_fft_values = fft_audio_data[: len(positive_freqs)]
    plt.subplot(2, 1, 1)
    plt.plot(
        positive_freqs, np.abs(positive_fft_values), label="Magnitude Napięcia (FFT)"
    )
    plt.title("Napięcie vs Częstotliwość")
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Amplituda Napięcia")
    plt.legend()

    # Plot the magnitude of the FFT of the current signal
    plt.subplot(2, 1, 2)
    plt.plot(
        positive_freqs,
        np.abs(np.fft.fft(current_signal)[: len(positive_freqs)]),
        label="Magnitude Prądu (FFT)",
        color="r",
    )
    plt.title("Prąd vs Częstotliwość")
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Amplituda Prądu")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main function to run the script
def main(file_path):
    sample_rate, audio_data = load_wav(file_path)
    freqs, current_signal = calculate_current(audio_data, impedance_data, sample_rate)
    plot_results(freqs, audio_data, current_signal)


# Run the script with the path to the WAV file
if __name__ == "__main__":
    wav_file = "output/sine_wave.wav"  # Replace with your file path
    main(wav_file)
