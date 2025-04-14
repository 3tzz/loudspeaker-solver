import numpy as np
from scipy.integrate import solve_ivp

from mechanical.signal_anal import analyse_steady_state_signal
from tools.audio.make_sound import generate_chord, normalize
from tools.plot.configs import Axis, Line, Plotter, Stem, Subplot
from tools.plot.multi_plotter import MultiPlotter


def simulate_ode_rk(t, freqs, amplitudes, m, k, c, initial_conditions, interval):
    """
    Simulates the motion of a mass-spring-damper system
    using the ODE Runge-Kutta method based on
    Newton's second law of motion."""
    sol = solve_ivp(
        ode_mass_spring_damper_definition,
        interval,
        initial_conditions,
        t_eval=t,
        args=(freqs, amplitudes, m, c, k),
    )
    return sol.y


def ode_mass_spring_damper_definition(t, y, frequencies, amplitudes, m, c, k):
    """
    Defines the system of first-order ODEs
    for the mass-spring-damper system with multi-harmonic external force.

    Parameters:
    - t: Current time
    - y: State vector [x, v] where x is displacement, v is velocity
    - F_func: Function representing the external force F(t)
    - m, c, k: System parameters

    Returns:
    - dydt: Array containing dx/dt and dv/dt
    """
    assert len(frequencies) == len(amplitudes)
    x, v = y
    F_t = sum(A * np.cos(2 * np.pi * f * t) for A, f in zip(amplitudes, frequencies))
    return [v, (F_t - c * v - k * x) / m]


# TODO: make numpyt array as func
# def ode_mass_spring_damper_definition(t, y, F_func, m, c, k):
#     """
#     Defines the system of first-order ODEs for the mass-spring-damper system.
#
#     Parameters:
#     - t: Current time
#     - y: State vector [x, v] where x is displacement, v is velocity
#     - F_func: Function representing the external force F(t)
#     - m, c, k: System parameters
#
#     Returns:
#     - dydt: Array containing dx/dt and dv/dt
#     """
#     x, v = y
#     F = F_func[t]
#     dxdt = v
#     dvdt = (F / m) - (c / m) * v - (k / m) * x
#
#     return [dxdt, dvdt]


def euler_mass_spring_damper(t, dt, F, m, c, k):
    """
    Simulates the motion of a mass-spring-damper system using the Euler method
    (a numerical integration method) based on Newton's second law of motion.

    The system is governed by the following differential equation:
    m * x''(t) + c * x'(t) + k * x(t) = F(t)
    where:
    - m: mass (kg)
    - c: damping coefficient (N·s/m)
    - k: spring constant (N/m)
    - x(t): displacement as a function of time
    - F(t): external force as a function of time

    The Euler method is used to approximate the solution to this second-order ODE.
    The method computes the displacement (x) and velocity (v) iteratively at each time step using the following update equations:
    - v[n] = v[n-1] + (F[n-1] / m - (c / m) * v[n-1] - (k / m) * x[n-1]) * dt
    - x[n] = x[n-1] + v[n-1] * dt

    Parameters:
    - t: Time array (time steps for simulation)
    - dt: Time resolution
    - F: External force array (F(t)) at each time step
    - m: Mass of the object (kg)
    - c: Damping coefficient (N·s/m)
    - k: Spring constant (N/m)

    Note:
    - This method works for any external force F(t) that is provided as an array
      where the force is defined at each time step.
    - The time step (dt) is assumed to be constant, so the method is appropriate for
      problems where the time resolution does not change over time.
    """
    assert len(t) == len(F)
    x = np.zeros(len(t))  # Initialize displacement array
    v = np.zeros(len(t))  # Initialize velocity array

    for n in range(1, len(t)):
        v[n] = v[n - 1] + (F[n - 1] / m - (c / m) * v[n - 1] - (k / m) * x[n - 1]) * dt
        x[n] = x[n - 1] + v[n - 1] * dt

    return x, v


def calculate_mechanical_parameters(fr: float, m: float, q_m: float):
    """
    Calculate mechanical parameters for the loudspeaker:
    - Stiffness (k)
    - Damping factor (d)

    :param fr: Resonance frequency in Hz
    :param m: Moving mass in kg
    :param q_m: Mechanical quality factor
    :return: Tuple (k, c)
    """

    # Compute stiffness
    k = (2 * np.pi * fr) ** 2 * m

    # Compute damping
    c = np.sqrt(m * k) / q_m

    return k, c


def main():
    # Generate signal
    duration = 2.0  # [s]
    fs = 2000  # [Hz]
    dt = int(duration * fs)  # Krok czasowy [s]
    dt = 0.001  # Time step (s)

    chord_frequencies = [440, 523.25, 659.25]  # A minor chord frequencies (A, C, E)
    amplitudes_weight = [20, 60, 100]

    t, signal = generate_chord(chord_frequencies, duration, fs, amplitudes_weight)
    signal = normalize(signal)
    frequencies_filtered, amplitudes_filtered = analyse_steady_state_signal(signal, fs)

    # Oscilation setup
    fr = 117  # resonance frequency[Hz]
    m = 0.0098  # mass [kg]
    q_m = 8.97  # mechanical quality factor
    q_t = 0.60  # total quality factor
    k, c = calculate_mechanical_parameters(
        fr, m, q_t
    )  # k (spring constant) [N/m] , c (damping) [N·s/m]

    # Euler simulation
    # x_euler, v_euler = euler_mass_spring_damper(t, dt, signal_filtered, m, c, k)
    x_euler, v_euler = euler_mass_spring_damper(t, dt, signal, m, c, k)

    # Euler analyse
    x_freq_filt, x_filt = analyse_steady_state_signal(x_euler, fs)
    v_freq_filt, v_filt = analyse_steady_state_signal(v_euler, fs)

    # ODE Runge-Kutta method
    x_oderk, v_oderk = simulate_ode_rk(
        t, frequencies_filtered, amplitudes_filtered, m, c, k, [0, 0], (0, duration)
    )
    # Euler analyse
    # x_freq_filt_oderk, x_filt_oderk = analyse_steady_state_signal(x_oderk, fs)
    # v_freq_filt_oderk, v_filt_oderk = analyse_steady_state_signal(v_oderk, fs)
    x_freq_filt_oderk, x_filt_oderk = analyse_steady_state_signal(
        x_oderk, fs, peak_height=0.000000001
    )
    v_freq_filt_oderk, v_filt_oderk = analyse_steady_state_signal(
        v_oderk, fs, peak_height=0.00001
    )

    # Plot the results
    plotter_config = Plotter(
        figsize=(10, 10), grid=True, tight_layout=True, sharex=False, sharey=False
    )
    plotter = MultiPlotter(config=plotter_config)
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Original Force",
                xlabel="Time [s]",
                ylabel="Amplitude [N]",
            ),
            chart_elements=[Line(t, signal, label="Force")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Force FFT",
                xlabel="Frequency [Hz]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Stem(frequencies_filtered, amplitudes_filtered, label="fft"),
            ],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Displacement",
                xlabel="Time [s]",
                ylabel="Displacement [m]",
            ),
            chart_elements=[Line(t, x_euler, label="Displacement")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Displacement FFT",
                xlabel="Frequency [Hz]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Stem(x_freq_filt, x_filt, label="fft"),
            ],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Velocity",
                xlabel="Time [s]",
                ylabel="Amplitude [m/s]",
            ),
            chart_elements=[Line(t, v_euler, label="Velocity")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Velocity FFT",
                xlabel="Frequency [Hz]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Stem(v_freq_filt, v_filt, label="fft"),
            ],
        )
    )
    plotter.plot()

    # Plot the results
    plotter_config = Plotter(
        figsize=(10, 10), grid=True, tight_layout=True, sharex=False, sharey=False
    )
    plotter = MultiPlotter(config=plotter_config)
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Original Force",
                xlabel="Time [s]",
                ylabel="Amplitude [N]",
            ),
            chart_elements=[Line(t, signal, label="Force")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Force FFT",
                xlabel="Frequency [Hz]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Stem(frequencies_filtered, amplitudes_filtered, label="fft"),
            ],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Displacement",
                xlabel="Time [s]",
                ylabel="Displacement [m]",
            ),
            chart_elements=[Line(t, x_oderk, label="Displacement")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Displacement FFT",
                xlabel="Frequency [Hz]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Stem(x_freq_filt_oderk, x_filt_oderk, label="fft"),
            ],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Velocity",
                xlabel="Time [s]",
                ylabel="Amplitude [m/s]",
            ),
            chart_elements=[Line(t, v_oderk, label="Velocity")],
        )
    )
    plotter.add_subplot(
        Subplot(
            axis_config=Axis(
                title="Velocity FFT",
                xlabel="Frequency [Hz]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Stem(v_freq_filt_oderk, v_filt_oderk, label="fft"),
            ],
        )
    )
    plotter.plot()


if __name__ == "__main__":
    main()
