import matplotlib.pyplot as plt
import numpy as np


# Function to calculate oscillating force
def oscillating_force(t, F0, f):
    return F0 * np.sin(2 * np.pi * f * t)  # Force oscillating at frequency f


# Runge-Kutta 4th-order method for displacement and velocity
def runge_kutta_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# Function for the system of equations with restoring force and damping
def system_of_equations(t, state, mass, F0, f, damping, k):
    velocity = state[0]
    displacement = state[1]

    # Calculate the oscillating force at time t
    force = oscillating_force(t, F0, f)

    # Calculate restoring force (Hooke's law)
    restoring_force = -k * displacement  # F = -kx

    # Total force (oscillating force + restoring force)
    total_force = force + restoring_force

    # Calculate acceleration (F = ma), adding damping
    acceleration = (total_force - damping * velocity) / mass

    # Return the derivatives
    return np.array([acceleration, velocity])


# Main function to simulate and plot
def main(F0, f, mass, damping, k, time_steps, dt):
    # Initial conditions: velocity and displacement
    initial_velocity = 0  # Starting at rest
    initial_displacement = 0  # Starting from initial position

    # Time vector
    time = np.linspace(0, time_steps * dt, time_steps)

    # Arrays to store velocity, displacement, acceleration, and force
    velocity = np.zeros(time_steps)
    displacement = np.zeros(time_steps)
    acceleration = np.zeros(time_steps)
    force = np.zeros(time_steps)

    # Set initial conditions
    velocity[0] = initial_velocity
    displacement[0] = initial_displacement

    # Use Runge-Kutta to solve the system
    for i in range(1, time_steps):
        # Get the state vector [velocity, displacement]
        state = np.array([velocity[i - 1], displacement[i - 1]])

        # Integrate using Runge-Kutta
        next_state = runge_kutta_step(
            lambda t, y: system_of_equations(t, y, mass, F0, f, damping, k),
            time[i - 1],
            state,
            dt,
        )

        # Extract updated velocity and displacement
        velocity[i] = next_state[0]
        displacement[i] = next_state[1]

        # Calculate force and acceleration
        force[i] = oscillating_force(time[i], F0, f)
        acceleration[i] = (force[i] - damping * velocity[i]) / mass

    # Plot the results
    plt.figure(figsize=(12, 10))

    # Plot displacement
    plt.subplot(4, 1, 1)
    plt.plot(time, displacement, label="Displacement (m)")
    plt.title("Displacement")
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [m]")

    # Plot velocity
    plt.subplot(4, 1, 2)
    plt.plot(time, velocity, label="Velocity (m/s)", color="r")
    plt.title("Velocity")
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")

    # Plot acceleration
    plt.subplot(4, 1, 3)
    plt.plot(time, acceleration, label="Acceleration (m/s²)", color="g")
    plt.title("Acceleration")
    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")

    # Plot force
    plt.subplot(4, 1, 4)
    plt.plot(time, force, label="Force (N)", color="b")
    plt.title("Oscillating Force")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")

    plt.tight_layout()
    plt.show()


# Parameters
F0 = 1.0  # Maximum force amplitude (in Newtons)
f = 440  # Frequency of oscillation in Hz (440 Hz corresponds to the A4 note)
mass = 0.0098  # Mass of voice coil in kg
damping = 0.02  # Damping coefficient (small value to prevent runaway oscillations)
k = 50  # Spring constant (N/m), this will control the oscillation range
time_steps = 1000  # Number of time steps
dt = 0.001  # Time step (1 ms)

# Run the simulation
main(F0, f, mass, damping, k, time_steps, dt)
