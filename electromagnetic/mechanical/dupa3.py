import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Parameters for the system
m = 0.01  # kg (mass)
c = 0.5  # N·s/m (damping)
k = 100  # N/m (spring constant)
f1 = 450  # Hz (frequency 1 of force)
f2 = 400  # Hz (frequency 2 of force)
A1 = 1  # Amplitude for frequency 1 (10 Hz)
A2 = 0.5  # Amplitude for frequency 2 (100 Hz)
omega1 = 2 * np.pi * f1  # Frequency 1 in radians per second
omega2 = 2 * np.pi * f2  # Frequency 2 in radians per second
dt = 0.001  # Time step (s)
T = 2  # Total simulation time (s)

# Time array
t = np.arange(0, T, dt)

# Arrays for displacement (x), velocity (v)
x = np.zeros(len(t))
v = np.zeros(len(t))

# External force F(t) with two frequencies and weighted amplitudes
F = A1 * np.cos(omega1 * t) + A2 * np.cos(omega2 * t)  # Combined external force

# Loop to solve the system
for n in range(1, len(t)):
    # Update velocity and displacement using the difference equations
    v[n] = v[n - 1] + (F[n - 1] / m - (c / m) * v[n - 1] - (k / m) * x[n - 1]) * dt
    x[n] = x[n - 1] + v[n - 1] * dt

# Plot the results
plt.figure(figsize=(10, 10))

# Plot displacement vs time
plt.subplot(4, 1, 1)
plt.plot(t, x, label="Displacement (x)")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.title("Mass-Spring-Damper System - Displacement")
plt.grid(True)

# Plot velocity vs time
plt.subplot(4, 1, 2)
plt.plot(t, v, label="Velocity (v)", color="g")
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.title("Mass-Spring-Damper System - Velocity")
plt.grid(True)

# Plot external force vs time
plt.subplot(4, 1, 3)
plt.plot(t, F, label="External Force (F)", color="r")
plt.xlabel("Time [s]")
plt.ylabel("Force [N]")
plt.title("External Force vs Time (Combined Frequencies)")
plt.grid(True)

# Calculate the FFT of displacement, velocity, and external force
n = len(t)  # Number of points in FFT
frequencies = np.fft.fftfreq(n, dt)  # Frequency axis
frequencies = np.fft.fftshift(frequencies)  # Shift zero frequency to center
fft_x = np.fft.fft(x)  # FFT of displacement
fft_v = np.fft.fft(v)  # FFT of velocity
fft_F = np.fft.fft(F)  # FFT of external force
fft_x = np.fft.fftshift(fft_x)  # Shift FFT for displacement
fft_v = np.fft.fftshift(fft_v)  # Shift FFT for velocity
fft_F = np.fft.fftshift(fft_F)  # Shift FFT for external force

# Plot the FFT magnitude
plt.subplot(4, 1, 4)
plt.plot(frequencies, np.abs(fft_x), label="FFT of Displacement (x)", color="b")
plt.plot(
    frequencies, np.abs(fft_v), label="FFT of Velocity (v)", color="g", linestyle="--"
)
plt.plot(
    frequencies, np.abs(fft_F), label="FFT of Force (F)", color="r", linestyle="--"
)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("Frequency Spectrum of Displacement, Velocity, and External Force")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# Find peaks in the FFT results
def print_fft_peaks(frequencies, fft_data, label):
    # Find the peaks in the FFT magnitude
    peaks, _ = find_peaks(np.abs(fft_data))
    peak_frequencies = frequencies[peaks]
    peak_magnitudes = np.abs(fft_data)[peaks]

    print(f"Peak Frequencies and Magnitudes for {label}:")
    for freq, mag in zip(peak_frequencies, peak_magnitudes):
        print(f"Frequency: {freq:.2f} Hz, Magnitude: {mag:.4f}")


# Print peaks for displacement, velocity, and force
print_fft_peaks(frequencies, fft_x, "Displacement (x)")
print_fft_peaks(frequencies, fft_v, "Velocity (v)")
print_fft_peaks(frequencies, fft_F, "External Force (F)")
