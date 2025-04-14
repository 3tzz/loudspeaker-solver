import matplotlib.pyplot as plt
import numpy as np


# 1. Method to generate acoustic wave (sine wave)
def generate_acoustic_wave(frequency, duration, sampling_rate):
    """
    Generate an acoustic wave (sine wave) with a given frequency, duration, and sampling rate.

    Args:
    - frequency (Hz)
    - duration (s)
    - sampling_rate (Hz)

    Returns:
    - time array
    - acoustic wave (sine wave)
    """
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    acoustic_wave = np.sin(2 * np.pi * frequency * t)
    return t, acoustic_wave


# 2. Method to convert acoustic wave to electric current for the voice coil
def convert_to_electric_current(acoustic_wave, gain_factor):
    """
    Convert the acoustic wave amplitude to electric current for the voice coil.

    Args:
    - acoustic_wave: The acoustic wave signal.
    - gain_factor: Proportionality constant to convert to current.

    Returns:
    - Electric current waveform
    """
    return gain_factor * acoustic_wave


# 3. Method to calculate the force on the coil based on the electric current
def calculate_force_on_coil(current_in_coil, Bl):
    """
    Calculate the force acting on the coil due to the magnetic field.

    Args:
    - current_in_coil: Electric current running through the voice coil.
    - Bl: Force factor of the voice coil (N/A).

    Returns:
    - Force on the coil (N)
    """
    return Bl * current_in_coil


# 4. Method to simulate the movement of the coil (displacement over time)
def simulate_coil_motion(force_on_coil, mass_of_coil, t):
    """
    Simulate the movement of the voice coil (displacement) over time using Newton's second law.

    Args:
    - force_on_coil: The force on the coil at each time step.
    - mass_of_coil: Mass of the voice coil (kg).
    - t: Time array.

    Returns:
    - Position (displacement) of the coil over time
    """
    # Initialize arrays for velocity and position
    velocity = np.zeros_like(t)
    position = np.zeros_like(t)

    # Time step for numerical integration (Euler method)
    dt = t[1] - t[0]

    # Numerical integration (Euler method) for velocity and position
    for i in range(1, len(t)):
        # Compute acceleration (F = m * a -> a = F / m)
        acceleration = force_on_coil[i - 1] / mass_of_coil

        # Update velocity and position
        velocity[i] = velocity[i - 1] + acceleration * dt
        position[i] = position[i - 1] + velocity[i - 1] * dt

    return position


# 5. Main method to run the full simulation
def run_simulation(frequency, duration, sampling_rate, gain_factor, Bl, mass_of_coil):
    """
    Run the full simulation pipeline for the loudspeaker model.

    Args:
    - frequency: Frequency of the acoustic wave in Hz.
    - duration: Duration of the simulation in seconds.
    - sampling_rate: Sampling rate in Hz.
    - gain_factor: Proportionality constant for converting acoustic wave to current.
    - Bl: Force factor of the voice coil (N/A).
    - mass_of_coil: Mass of the voice coil (kg).

    Returns:
    - Time array, displacement of the coil (position).
    """
    # Generate acoustic wave
    t, acoustic_wave = generate_acoustic_wave(frequency, duration, sampling_rate)

    # Convert acoustic wave to electric current for the voice coil
    current_in_coil = convert_to_electric_current(acoustic_wave, gain_factor)

    # Calculate the force on the coil
    force_on_coil = calculate_force_on_coil(current_in_coil, Bl)

    # Simulate the coil motion (displacement over time)
    position = simulate_coil_motion(force_on_coil, mass_of_coil, t)

    return t, position


# 6. Visualization method (optional)
def plot_simulation(t, position):
    """
    Plot the displacement of the voice coil over time.

    Args:
    - t: Time array.
    - position: Displacement of the coil over time.
    """
    plt.plot(t, position)
    plt.title("Position of the Coil (Displacement)")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")
    plt.show()


# Main execution: running the simulation
if __name__ == "__main__":
    # Parameters for the simulation
    frequency = 1000  # Frequency in Hz (1kHz)
    duration = 0.01  # Duration in seconds
    sampling_rate = 44100  # Sampling rate in Hz
    gain_factor = 0.01  # Gain factor for converting acoustic wave to current
    Bl = 8.26  # Force factor in N/A
    mass_of_coil = 0.05  # Mass of the voice coil in kg

    # Run the simulation
    t, position = run_simulation(
        frequency, duration, sampling_rate, gain_factor, Bl, mass_of_coil
    )

    # Plot the results
    plot_simulation(t, position)
