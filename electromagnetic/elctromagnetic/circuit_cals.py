import json
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav

# Voice coil parameters
impedance = 8  # Ohms (not used directly)
inductance = 0.53e-3  # H (constant)
resistance = 6.1  # Ohms


# Load WAV file
def load_wav(file_path):
    sample_rate, audio_data = wav.read(file_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)  # Convert stereo to mono
    return sample_rate, audio_data


# Convert audio signal to voice coil current and inductor voltage
def audio_to_circuit(audio_data, sample_rate, resistance, inductance):
    time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))

    # Calculate current (Ohm’s law: I = V / R)
    current_signal = audio_data / resistance

    # Compute voltage across the inductor (V_L = L * dI/dt) inductance
    dt = 1 / sample_rate
    di_dt = np.diff(current_signal) / dt
    voltage_inductor = inductance * di_dt

    # Pad to match original signal length
    voltage_inductor = np.pad(voltage_inductor, (1, 0), "constant")

    return time, current_signal, voltage_inductor


# Function to extract data for a given time range
def extract_time_range(time, data, time_range: tuple[float] | None):
    """Extracts data within a specified time range. If None, returns full range."""
    if not time_range:
        return time, data
    assert isinstance(time_range, tuple)

    t_start = time_range[0]
    t_end = time_range[-1]
    mask = (time >= t_start) & (time <= t_end)
    return time[mask], data[mask]


# Plot results with a zoomed-in time range
def plot_results(time, audio_data, current_signal, voltage_inductor, time_range=None):
    plt.figure(figsize=(12, 8))

    # Plot raw audio signal (zoomed)
    plt.subplot(3, 1, 1)
    t_zoom, audio_zoom = extract_time_range(time, audio_data, time_range)
    plt.plot(t_zoom, audio_zoom, label="Raw Audio Signal", color="g")
    plt.title(f"Loaded Audio Signal {time_range}s")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()

    # Plot current response (zoomed)
    plt.subplot(3, 1, 2)
    t_zoom, current_zoom = extract_time_range(time, current_signal, time_range)
    plt.plot(t_zoom, current_zoom, label="Current (I)")
    plt.title(f"Voice Coil Current {time_range}s")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.legend()

    # Plot inductor voltage response (zoomed)
    plt.subplot(3, 1, 3)
    t_zoom, voltage_zoom = extract_time_range(time, voltage_inductor, time_range)
    plt.plot(t_zoom, voltage_zoom, label="Inductor Voltage (V_L)", color="r")
    plt.title(f"Inductor Voltage Response {time_range}s")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Save results to a JSON file
def save_to_json(
    time,
    sample_rate,
    audio_data,
    current_signal,
    voltage_inductor,
    output_path,
    filename="results.json",
):
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Define the full file path
    file_path = os.path.join(output_path, filename)

    data = {
        "time": time.tolist(),
        "sample_rate": sample_rate,
        "audio_signal": audio_data.tolist(),
        "current_signal": current_signal.tolist(),
        "voltage_inductor": voltage_inductor.tolist(),
    }

    with open(file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Results saved to {file_path}")


# Main function
def main(file_path, output_path):
    sample_rate, audio_data = load_wav(file_path)
    time, current_signal, voltage_inductor = audio_to_circuit(
        audio_data, sample_rate, resistance, inductance
    )

    # Save results to JSON
    save_to_json(
        time, sample_rate, audio_data, current_signal, voltage_inductor, output_path
    )

    # Plot results with a zoomed-in time range
    plot_results(
        time, audio_data, current_signal, voltage_inductor, time_range=(0.01, 0.06)
    )


# Run script
if __name__ == "__main__":
    wav_file = "output/sine_wave.wav"
    output_directory = "output"
    main(wav_file, output_directory)
