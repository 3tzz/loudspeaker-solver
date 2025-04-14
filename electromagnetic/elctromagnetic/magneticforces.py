import json

import matplotlib.pyplot as plt
import numpy as np


# Function to load the results from a JSON file
def load_json_results(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def calculate_constant_magnetic_field(Bl, l):
    """Constant magnetic field."""
    return Bl / l


def calculate_lorentz_force(I, B, l):
    """Siła Lorentza F = B * l * I"""
    return B * l * np.array(I)


def simulate_motion(current_signal, magnetic_field, coil_length, mass, sample_rate):
    # Inicjalizujemy prędkość, przemieszczenie i przyspieszenie
    velocity = np.zeros_like(current_signal)
    displacement = np.zeros_like(current_signal)
    acceleration = np.zeros_like(current_signal)
    force = np.zeros_like(current_signal)

    # Krok czasowy (1/sampling rate)
    dt = 1 / sample_rate

    # Symulacja ruchu w czasie z użyciem integracji Velocity-Verlet
    for i in range(1, len(current_signal)):
        # Obliczanie siły na podstawie siły Lorentza
        force[i] = calculate_lorentz_force(
            current_signal[i], magnetic_field, coil_length
        )

        # Obliczanie przyspieszenia za pomocą II zasady Newtona (a = F/m)
        acceleration[i] = force[i] / mass

        # Aktualizacja prędkości przy użyciu półkroku Velocity-Verlet
        velocity[i] = (
            velocity[i - 1] + 0.5 * (acceleration[i] + acceleration[i - 1]) * dt
        )

        # Aktualizacja przemieszczenia przy użyciu pełnego kroku Velocity-Verlet
        displacement[i] = displacement[i - 1] + velocity[i] * dt

    return displacement, velocity, acceleration, force


def plot_motion(time, displacement, velocity, acceleration, force, current_signal):
    plt.figure(figsize=(12, 10))

    # Plot current signal
    plt.subplot(5, 1, 1)
    plt.plot(time, current_signal, label="Current Signal (A)", color="m")
    plt.title("Current Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.legend()

    # Plot displacement
    plt.subplot(5, 1, 2)
    plt.plot(time, displacement, label="Displacement (m)")
    plt.title("Displacement")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.legend()

    # Plot velocity
    plt.subplot(5, 1, 3)
    plt.plot(time, velocity, label="Velocity (m/s)", color="r")
    plt.title("Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()

    # Plot acceleration
    plt.subplot(5, 1, 4)
    plt.plot(time, acceleration, label="Acceleration (m/s²)", color="g")
    plt.title("Acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    plt.legend()

    # Plot Lorentz force
    plt.subplot(5, 1, 5)
    plt.plot(time, force, label="Lorentz Force (N)", color="b")
    plt.title("Lorentz Force")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.legend()

    plt.tight_layout()
    plt.show()


# Main function to demonstrate loading and displaying the results
def main(json_file: str, params: dict, constant_field=True):
    signal_params = load_json_results(json_file)
    I = signal_params["current_signal"]
    B = calculate_constant_magnetic_field(params["Bl"], params["air_gap"])

    displacement, velocity, acceleration, force = simulate_motion(
        I, B, params["HVC"], params["moving_mass"], signal_params["sample_rate"]
    )

    print(f"Magnetic field: {B} T")
    print(f"Displacement: {displacement[-1]} m")
    print(f"Velocity: {velocity[-1]} m/s")
    print(f"Acceleration: {acceleration[-1]} m/s²")

    # Create time vector
    time = np.linspace(0, len(I) / signal_params["sample_rate"], len(I))

    # Plot the results
    plot_motion(time, displacement, velocity, acceleration, force, I)


if __name__ == "__main__":
    params = {
        "Z": 8,  # Nominalna impedancja w Ohmach
        "RE": 6.1,  # Opór DC w Ohmach
        "LE": 0.53e-3,  # Indukcyjność w Henrych (H)
        "VC_diameter": 38e-3,  # Średnica cewki głosnika w metrach
        "air_gap": 12e-3,  # Wysokość nawinięcia w metrach
        "Bl": 8.26,  # Siła elektromotoryczna w N/A
        "HVC": 8e-3,
        "motor_constant": 3.34,  # Stała silnika w N/√W
        "moving_mass": 9.8e-3,  # Masa poruszająca się cewki (9.8g -> 0.0098 kg)
    }
    json_current = "output/results.json"  # Path to your JSON file
    constant_field = True
    main(json_current, params, constant_field=constant_field)
