# # pylint: disable=missing-module-docstring
# from pathlib import Path
# import argparse
# from numpy import numpy

# from boomspeaver.loudspeaker.schema import Loudspeaker
# from boomspeaver.tools.data import load_numpy_file, save_numpy_file

# def fourier_analytical() -> np.ndarray:




import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from functools import cached_property
from abc import ABC, abstractmethod


from boomspeaver.tools.signal.make_sound import generate_chord, generate_time_domain
from boomspeaver.tools.signal.signal import fft_analysis, stft, stft_generator, detect_peaks
from boomspeaver.tools.data import load_numpy_file
from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.plot.configs import Axis, Line, Plotter, Points, Subplot
from boomspeaver.tools.plot.multi_plotter import MultiPlotter

def test() -> None:

    # Define chord frequencies for A minor chord (A, C, E)
    # Frequencies of the A minor chord notes: 440 Hz (A), 523.25 Hz (C), 659.25 Hz (E)
    sampling_rate = 10000  # Sampling rate in Hz
    duration = 2.0  # Duration of the signal in seconds
    chord_frequencies = [440, 523.25, 659.25]  # A minor chord frequencies (A, C, E)
    chord_amplitudes = [1.0, 2.0, 0.5]
    chord_phases = [0.0, np.pi / 4, np.pi / 2]

    # Generate the chord signal
    t = generate_time_domain(sampling_rate=sampling_rate, duration=duration)
    signal = generate_chord(
        frequencies=chord_frequencies,
        amplitudes=chord_amplitudes,
        duration=duration,
        fs=sampling_rate,
        signal_norm=True,
        silence=False,
    )





    # Parameters
    m = 1.0          # mass
    # c = 0.2          # damping coefficient
    c = 0          # damping coefficient
    # k = 4.0          # stiffness
    k = 1.0          # stiffness

    # Derived parameters
    omega0 = np.sqrt(k / m)                   # natural frequency
    zeta = c / (2 * np.sqrt(k * m))           # damping ratio
    omega_d = omega0 * np.sqrt(1 - zeta**2)   # damped natural frequency

    # Driving forces: list of (A_i, f_i, phi_i)
    # forces = [
    #     (1.0, 0.5, 0.0),           # amplitude=1, freq=0.5Hz, phase=0
    #     (0.5, 1.2, np.pi / 4),     # amplitude=0.5, freq=1.2Hz, phase=π/4
    #     (0.3, 2.0, np.pi / 2),     # amplitude=0.3, freq=2.0Hz, phase=π/2
    # ]
    assert len(chord_frequencies) == len(chord_amplitudes)
    forces = [(ai,fi,phi) for ai, fi, phi in zip(chord_amplitudes,chord_frequencies,chord_phases)]

    # Compute particular solution components
    def particular_solution(t, A, f, phi):
        omega = 2 * np.pi * f
        # Amplitude factor B_i
        denom = np.sqrt((omega0**2 - omega**2)**2 + (2 * zeta * omega0 * omega)**2)
        B = (A / m) / denom
        # Phase shift psi_i
        psi = np.arctan2(2 * zeta * omega0 * omega, omega0**2 - omega**2)
        return B * np.cos(omega * t + phi - psi), B, psi

    # Sum particular solutions and store B_i, psi_i for initial condition calc
    x_p_total = np.zeros_like(t)
    B_list = []
    psi_list = []
    omega_list = []
    phi_list = []
    A_list = []

    for (A_i, f_i, phi_i) in forces:
        x_pi, B_i, psi_i = particular_solution(t, A_i, f_i, phi_i)
        x_p_total += x_pi
        B_list.append(B_i)
        psi_list.append(psi_i)
        omega_list.append(2 * np.pi * f_i)
        phi_list.append(phi_i)
        A_list.append(A_i)

    # Initial conditions
    x0 = 0.0     # initial displacement
    v0 = 0.0     # initial velocity

    # Evaluate particular solution and its derivative at t=0
    x_p_0 = 0.0
    dx_p_0 = 0.0
    for B_i, psi_i, omega_i, phi_i in zip(B_list, psi_list, omega_list, phi_list):
        x_p_0 += B_i * np.cos(phi_i - psi_i)
        dx_p_0 += -B_i * omega_i * np.sin(phi_i - psi_i)

    # Compute constants C1, C2 for homogeneous solution
    C1 = x0 - x_p_0
    C2 = (v0 - dx_p_0 + zeta * omega0 * C1) / omega_d

    # Homogeneous solution
    x_h = np.exp(-zeta * omega0 * t) * (C1 * np.cos(omega_d * t) + C2 * np.sin(omega_d * t))

    # Total solution
    x_total = x_h + x_p_total

    # Perform FFT analysis
    freqs_org, amplitudes_org = fft_analysis(signal, sampling_rate)
    freqs_total, amplitudes_total = fft_analysis(x_total, sampling_rate)

    idxes_org, filtered_signal_org = detect_peaks(amplitudes_org)
    idxes_total, filtered_signal_total = detect_peaks(amplitudes_total)

    filtered_freqs_org = freqs_org[idxes_org]
    filtered_freqs_total = freqs_total[idxes_total]

    print(filtered_freqs_org)
    print(filtered_freqs_total)

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
                Points(
                    filtered_freqs_org,
                    filtered_signal_org,
                    label="Detected Peaks",
                    marker="o",
                    color="r",
                ),
                Line(freqs_org, amplitudes_org, label="fft"),
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
                title="Damped Forced Oscillator with Multiple Cosine Forces",
                xlabel="Time [s]",
                ylabel="Amplitude",
            ),
            chart_elements=[
                Line(t, x_total, label="Total solution $x(t)$"),
                Line(t, x_h, label="Transient (homogeneous)"),
                Line(t, x_p_total, label="Steady-state (particular)"),
                ],
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
                Points(
                    filtered_freqs_total,
                    filtered_signal_total,
                    label="Detected Peaks",
                    marker="o",
                    color="r",
                ),
                Line(freqs_total, amplitudes_total, label="fft"),
            ],
        )
    )
    plotter.plot()

def homogeneous_solution(t: np.ndarray, omega0: float, zeta: float, x0: float, v0: float):
    """
    Computes the homogeneous (natural) solution of the damped harmonic oscillator,
    i.e. the system's response due to initial displacement and velocity, without any external force.

    The homogeneous solution depends on the damping ratio ζ:

    - **Underdamped (ζ < 1)**:
        u_h(t) = e^(−ζω₀t) · [x₀ · cos(ω_d t) + ((v₀ + ζω₀x₀) / ω_d) · sin(ω_d t)]
        where ω_d = ω₀ · sqrt(1 − ζ²)

    - **Critically damped (ζ = 1)**:
        u_h(t) = (x₀ + (v₀ + ω₀x₀)t) · e^(−ω₀t)

    - **Overdamped (ζ > 1)**:
        u_h(t) = A · e^(r₁t) + B · e^(r₂t)
        where:
            r₁ = −ω₀(ζ − √(ζ² − 1))
            r₂ = −ω₀(ζ + √(ζ² − 1))
            A = (v₀ − r₂x₀) / (r₁ − r₂)
            B = x₀ − A

    This natural response decays over time due to damping and vanishes as t → ∞.
    """
    if zeta < 1:  # Underdamped
        omega_d = omega0 * np.sqrt(1 - zeta**2)
        A = x0
        B = (v0 + zeta * omega0 * x0) / omega_d
        return np.exp(-zeta * omega0 * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    elif zeta == 1:  # Critically damped
        A = x0
        B = v0 + omega0 * x0
        return (A + B * t) * np.exp(-omega0 * t)

    else:  # Overdamped
        r1 = -omega0 * (zeta - np.sqrt(zeta**2 - 1))
        r2 = -omega0 * (zeta + np.sqrt(zeta**2 - 1))
        A = (v0 - r2 * x0) / (r1 - r2)
        B = x0 - A
        return A * np.exp(r1 * t) + B * np.exp(r2 * t)

def particular_solution(t: np.ndarray, A: float, f: float, phi: float,
                        m: float, omega0: float, zeta: float) -> tuple[np.ndarray, float, float]:
    """
    Computes the steady-state response of a damped harmonic oscillator
    to a sinusoidal driving force A * cos(2πf * t + φ).

    The particular solution has the form:
        u_p(t) = B * cos(2πf * t + φ - ψ)

    where:
        B  = (A / m) / sqrt((ω₀² - ω²)² + (2ζω₀ω)²)
        ψ  = arctan2(2ζω₀ω, ω₀² - ω²)
        ω  = 2πf       — angular frequency of the driving force
        ω₀ = sqrt(k / m) — natural frequency of the system
        ζ  = c / (2√(km)) — damping ratio

    This solution represents a phase-shifted cosine with frequency matching the driving force,
    scaled by a frequency-dependent amplitude factor.
    """
    omega = 2 * np.pi * f
    denom = np.sqrt((omega0**2 - omega**2)**2 + (2 * zeta * omega0 * omega)**2)

    B = (A / m) / denom
    psi = np.arctan2(2 * zeta * omega0 * omega, omega0**2 - omega**2)

    u_i = B * np.cos(omega * t + phi - psi)
    return u_i, B, psi


class Solver:
    def solve(self, time: np.ndarray, initial_displacement: float, initial_velocity: float):
        raise NotImplementedError

class HomogeneousOscillationSolver:
    def __init__(self, system: "HarmonicOscillatorSystem"):
        self.system = system
        dtype = system.damping_type

        if dtype == "overdamped":
            self.solver = OverdampedSolver(system)
        elif dtype == "critical":
            self.solver = CriticallyDampedSolver(system)
        elif dtype == "underdamped":
            self.solver = UnderdampedSolver(system)
        else:
            raise NotImplementedError

    def solve(self, time: np.ndarray, x0: float, v0: float):
        return self.solver.solve(time, x0, v0)

class OverdampedSolver(Solver):
    """
    Computes the transient (homogeneous) response x(t) and ẋ(t) for overdamped motion.

    The response is:
        x(t) = A₁ ⋅ e^(−ω₊ ⋅ t) + A₂ ⋅ e^(−ω₋ ⋅ t)
        ẋ(t) = −A₁ ⋅ ω₊ ⋅ e^(−ω₊ ⋅ t) − A₂ ⋅ ω₋ ⋅ e^(−ω₋ ⋅ t)
    where:
        ω₊ = −(−Γ/2 + sqrt((Γ/2)² − ω₀²))
        ω₋ = −(−Γ/2 − sqrt((Γ/2)² − ω₀²))
    where A₁ and A₂ depends on initial condition.
        x(0) = A₁ + A₂ = x₀
        ẋ(0) = −A₁ ⋅ ω₊ − A₂ ⋅ ω₋ = v₀
    """
    def __init__(self, system: "HarmonicOscillatorSystem"):
        self.system = system
        assert self.system.damping_type.lower() == "overdamped"

    @cached_property
    def delta(self):
        """
        sqrt((Γ/2)² − ω₀²)
        """
        return np.sqrt(self.system.total_c_by_2**2 - self.system.omega0**2)

    @cached_property
    def omega_plus(self):
        """
        ω₊ = −(−Γ/2 + sqrt((Γ/2)² − ω₀²))
        """
        return self.system.total_c_by_2 - self.delta

    @cached_property
    def omega_minus(self):
        """
        ω₋ = −(−Γ/2 − sqrt((Γ/2)² − ω₀²))
        """
        return self.system.total_c_by_2 + self.delta

    def calculate_initial_constants(self,initial_displacement:float, initial_velocity) -> tuple[float, float]:
        """
        Calculate A₁ and A₂:
            x(0) = A₁ + A₂ = x₀
            ẋ(0) = −A₁ ⋅ ω₊ − A₂ ⋅ ω₋ = v₀
        """
        A = np.array([[1, 1],
                      [-self.omega_plus, -self.omega_minus]])
        b = np.array([initial_displacement, initial_velocity])
        A1, A2 = np.linalg.solve(A, b)
        return A1, A2

    def calculate_constant_part(self, time: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        """
        The time dependent constants:
            e^(−ω₊ ⋅ t)
            e^(−ω₋ ⋅ t))
        """
        e1 = np.exp(-self.omega_plus * time)
        e2 = np.exp(-self.omega_minus * time)
        return e1, e2

    def solve(self, time: np.ndarray, initial_displacement: float, initial_velocity: float) -> tuple[np.ndarray, np.ndarray]:
        """
        The calculate response:
            x(t) = A₁ ⋅ e^(−ω₊ ⋅ t) + A₂ ⋅ e^(−ω₋ ⋅ t)
            ẋ(t) = −A₁ ⋅ ω₊ ⋅ e^(−ω₊ ⋅ t) − A₂ ⋅ ω₋ ⋅ e^(−ω₋ ⋅ t)
        """
        A1, A2 = self.calculate_initial_constants(initial_displacement=initial_displacement, initial_velocity=initial_velocity)

        e1, e2 = self.calculate_constant_part(time)

        displacement = A1 * e1 + A2 * e2
        velocity = -A1 * self.omega_plus * e1 - A2 * self.omega_minus * e2
        return displacement, velocity

class CriticallyDampedSolver(Solver):
    """
    Computes the transient (homogeneous) response x(t) and ẋ(t) for critically damped motion.

    The response is:
        x(t) = (A + B⋅t) ⋅ e^(−Γ/2 ⋅ t)
        ẋ(t) = [B − (Γ/2)⋅(A + B⋅t)] ⋅ e^(−Γ/2 ⋅ t)

    where:
        A₁ and A₂ depend on initial conditions:
            x(0) = A = x₀
            ẋ(0) = B − (Γ/2)⋅A = v₀
    """
    def __init__(self, system: "HarmonicOscillatorSystem"):
        self.system = system
        assert self.system.damping_type.lower() == "critical"

    def calculate_initial_constants(self, initial_displacement: float, initial_velocity: float) -> tuple[float, float]:
        """
        Solve for A and B using:
            x(0) = A = x₀
            ẋ(0) = B − (Γ/2)⋅A = v₀
        """
        A = initial_displacement
        B = initial_velocity + self.system.total_c_by_2 * A
        return A, B

    def calculate_constant_part(self, time: np.ndarray) -> np.ndarray:
        """
        Compute e^(−Γ/2 ⋅ t)
        """
        return np.exp(-self.system.total_c_by_2 * time)

    def solve(self, time: np.ndarray, initial_displacement: float, initial_velocity: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the response:
            x(t) = (A + B⋅t) ⋅ e^(−Γ/2 ⋅ t)
            ẋ(t) = [B − (Γ/2)⋅(A + B⋅t)] ⋅ e^(−Γ/2 ⋅ t)
        """
        A1, A2 = self.calculate_initial_constants(initial_displacement, initial_velocity)
        decay = self.calculate_constant_part(time)

        displacement = (A1 + A2 * time) * decay
        velocity = (A2 - self.system.total_c_by_2 * (A1 + A2 * time)) * decay

        return displacement, velocity

class UnderdampedSolver(Solver):
    """
    Computes the transient (homogeneous) response x(t) and ẋ(t) for underdamped motion.

    The response is:
        x(t) = A ⋅ e^(−Γ/2 ⋅ t) ⋅ cos(ω₁ t − β)
        ẋ(t) = −A ⋅ e^(−Γ/2 ⋅ t) ⋅ [ω₁ ⋅ sin(ω₁ t − β) + Γ/2 ⋅ cos(ω₁ t − β)]

    where:
        A, β depend on initial conditions:
            x(0) = A ⋅ cos(−β)
            ẋ(0) = −A ⋅ [ω₁ ⋅ sin(−β) + Γ/2 ⋅ cos(−β)]
    """

    def __init__(self, system: "HarmonicOscillatorSystem"):
        self.system = system
        assert self.system.damping_type.lower() == "underdamped"

    def calculate_initial_constants(self, initial_displacement: float, initial_velocity: float) -> tuple[float, float]:
        """
        Solve for A and β using:
            x(0) = A ⋅ cos(β)
            ẋ(0) = −A ⋅ [ω₁ ⋅ sin(β) + Γ/2 ⋅ cos(β)]
        """
        Γ2 = self.system.total_c_by_2
        ω1 = self.system.omega1

        β = np.arctan2(
            ω1 * initial_displacement,
            Γ2 * initial_displacement + initial_velocity
        )

        A = initial_displacement / np.cos(β)

        return A, β

    def calculate_constant_part(self, time: np.ndarray) -> np.ndarray:
        """
        Compute e^(−Γ/2 ⋅ t)
        """
        return np.exp(-self.system.total_c_by_2 * time)

    def solve(self, time: np.ndarray, x0: float, v0: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Return x(t), ẋ(t) for underdamped case
            x(t) = A ⋅ e^(−Γ/2 ⋅ t) ⋅ cos(ω₁ t − β)
            ẋ(t) = −A ⋅ e^(−Γ/2 ⋅ t) ⋅ [ω₁ ⋅ sin(ω₁ t − β) + Γ/2 ⋅ cos(ω₁ t − β)]
        """
        A, β = self.calculate_initial_constants(x0, v0)
        e1= self.calculate_constant_part(time)

        phase = self.system.omega1 * time - β

        displacement = A * e1 * np.cos(phase)
        velocity = -A * e1 * (self.omega1 * np.sin(phase) + self.system.total_c_by_2 * np.cos(phase))

        return displacement, velocity

class ParticularFrequency(Solver):
    """
        m·ẍ(t) + c·ẋ(t) + k·x(t) = Fⱼ·cos(ωⱼ·t + φⱼ)
    Computes the particular (steady-state) solution of the driven damped harmonic oscillator:
        x(t) = A ⋅ cos(ω t + δ - δ_res)
        ẋ(t) = -A ⋅ ω ⋅ sin(ω t + δ - δ_res)
    where:
        A, β depend on initial conditions amplitude and phase shift.
    """

    def __init__(self, system: "HarmonicOscillatorSystem"):
        self.system = system

    def calculate_amplitude(self, frequency: float, amplitude: float) -> float:
        """
        Calculate amplitude (A) for solution:
            x(t) = A ⋅ cos(ω t + δ - δ_res)
        The amplitude A is given by:
            A = (F₀ / m) / sqrt((ω₀² − ω²)² + (bω / m)²)
        """
        omega = (self.system.omega0**2 - frequency**2)**2
        amp = amplitude/self.system.mass
        damp = ((self.system.total_c * frequency))**2
        return  amp / np.sqrt(omega + damp)

    def calculate_phase(self, frequency: np.ndarray) -> float:
        """
        Calculate phase (δ) for solution:
            x(t) = A ⋅ cos(ω t + δ - δ_res)
        The phase shift δ  is given by:
            δ  = arctan( (b/m) ω / (ω₀² − ω²) )
        """
        numerator = (self.system.total_c)*frequency
        denominator = (self.system.omega0**2 - frequency**2)
        return np.arctan2(numerator, denominator)

    def solve(self, time: np.ndarray, frequency: float, amplitude: float, phase: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Driving force
            Fⱼ·cos(ωⱼ·t + φⱼ)
        Computes the particular (steady-state) solution of the driven damped harmonic oscillator:
            x(t) = A ⋅ cos(ω t + δ - δ_res)
            ẋ(t) = -A ⋅ ω ⋅ sin(ω t + δ - δ_res)
        """

        amplitude = self.calculate_amplitude(frequency, amplitude)
        angle = self.calculate_phase(frequency)

        span = frequency*time + angle - phase

        displacement = amplitude * np.cos(span)
        velocity = -amplitude * self.omega * np.sin(span)

        return displacement, velocity


class HarmonicOscillationSystem:
    """
    Solve the equation of motion for a linear damped harmonic oscillator with multiple harmonic external forces:
    m·ẍ(t) + c·ẋ(t) + k·x(t) = Σⱼ Fⱼ·cos(ωⱼ·t + φⱼ)
    https://phys.libretexts.org/Bookshelves/Classical_Mechanics/Variational_Principles_in_Classical_Mechanics_(Cline)/03%3A_Linear_Oscillators/3.06%3A_Sinusoidally-driven_linearly-damped_linear_oscillator
    """
    def __init__(self, mass: float, damping_coefficient: float, stiffness: float)->None:
        assert mass == 0
        self.m = mass
        self.c = damping_coefficient
        self.k = stiffness
        # self.omega0 = np.sqrt(self.k / self.m)  # natural frequency
        # self.zeta = self.c / (2 * np.sqrt(self.k * self.m)) # damping ratio
        # self.omega_d = self.omega0 * np.sqrt(1 - self.zeta**2) # damped natural frequency

    @cached_property
    def omega0(self):
        return np.sqrt(self.k / self.m) #natural frequency

    @cached_property
    def total_c(self):
        return self.c / self.m #total damping parameter

    @cached_property
    def total_c_by_2(self):
        return self.total_c/2

    @cached_property
    def omega1(self):
        return np.sqrt(self.omega0**2 - (self.total_c_by_2)**2) #damped natural frequency

    @cached_property
    def damping_type(self):
        square_omega1=self.omega1**2
        if square_omega1 > 0:
            return "underdamped"
        elif square_omega1 == 0:
            return "critically"
        elif square_omega1 < 0:
            return "overdamped"
        else:
            NotImplementedError

    def _get_underdamped_solution(self, time_domain: np.ndarray, initial_displacement: float, initial_velocity: float):
        """
        Computes the transient (homogeneous) response x(t) and ẋ(t) for underdamped motion.

        The response is based on the general solution for underdamped harmonic motion:
            x(t) = A ⋅ e^(−Γ/2 ⋅ t) ⋅ cos(ω₁ ⋅ t − β)
            ẋ(t) = −A ⋅ e^(−Γ/2 ⋅ t) ⋅ [ω₁ ⋅ sin(ω₁ ⋅ t − β) + Γ/2 ⋅ cos(ω₁ ⋅ t − β)]

        where A and beta depends on initial condition.
        """

        A, beta = self.compute_homogeneous_constants(initial_displacement, initial_velocity)
        envelope = np.exp(-0.5 * self.total_c * time_domain)
        phase = self.omega1 * time_domain - beta

        displacement = A * envelope * np.cos(phase)
        velocity = -A * envelope * (self.omega1 * np.sin(phase) + 0.5 * self.total_c * np.cos(phase))
        return displacement, velocity

    def _get_complementary_solution(self, initial_displacement: float, initial_velocity: float):
        """"""
        #TODO add time

        time_domain=None
        if self.type == "underdamped":
            return self._get_underdamped_solution(time_domain, initial_displacement, initial_velocity)
        else:
            raise NotImplementedError


    def solve_steady_state():
        pass

    def solve(self, frequencies: np.ndarray, amplitudes: np.ndarray, phases: np.ndarray, initial_displacement: float = 0.0, initial_velocity: float = 0.0):
        complementary_solution=None
        particular_solution=None
        # solution=homogeneous_solution+particular_solution
        solution=None
        return solution







def fourier_analytical(: np.ndarray, mass: float, damping_coefficient: float, stiffness: float, init_x: float=0.0, init_v: float=0.0):
    
    pass

def get_frame(frame: np.ndarray):
    return np.abs(frame), np.angle(frame)

def frame_generator(freqs: np.ndarray, spec: np.ndarray, threshold: float):
    """
    Yields (frequencies, amplitudes, phases) from spectrogram frames
    where magnitude exceeds threshold at peaks.
    """
    assert len(spec.shape) == 2
    assert len(freqs) == spec.shape[0]
    for i_spec in spec.T:
        magnitude, angle = get_frame(i_spec)
        peaks, _ = find_peaks(magnitude, height=threshold)
        if peaks.size > 0:
            yield freqs[peaks], magnitude[peaks], angle[peaks]
        else:
            yield np.array([0.0]), np.array([0.0]), np.array([0.0])


def main(
    input_signal_path: Path,
    loudspeaker_params: Loudspeaker,
    output_path: Path,
    n_fft: int = 2048,
    win_length: int = 2048,
    hop_length: int = 512,
    sampling_rate: int = 48000,
    amplitude_frequency_threshold = 0.1,
    ) -> None:

    assert isinstance(input_signal_path, Path)
    assert input_signal_path.exists()
    assert input_signal_path.suffix == ".npy"
    assert isinstance(output_path, Path)
    assert isinstance(loudspeaker_params, Loudspeaker)
    assert isinstance(sampling_rate, int) and sampling_rate >= 0

    signal = load_numpy_file(input_signal_path)
    assert len(signal.shape) == 1
    t = generate_time_domain(sampling_rate=sampling_rate, n_samples=len(signal))


    freqs, spec = stft(
        signal=signal,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )

    print(len(freqs))
    print(spec.shape[0])
    assert len(freqs) == spec.shape[0]

    for i, (frequency, amplitude, phase) in enumerate(frame_generator(freqs, spec, threshold=amplitude_frequency_threshold)):
        print(f"Frame {i} - peaks: {frequency}")
    raise


    n_frames = spec.shape[1]
    frame_time = np.arange(win_length) / sampling_rate  # time axis for a single frame

    # Calculate full length of reconstructed signal
    reconstructed_length = hop_length * (n_frames - 1) + win_length
    reconstructed_signal = np.zeros(reconstructed_length)
    overlap_counter = np.zeros(reconstructed_length)
    for i, i_spec in enumerate(spec.T):

        phases = np.angle(i_spec)
        phase=phases[peaks]

        # fourier_analytical(
        #     spec=i_spec,
        # )


        frame_signal = np.zeros_like(frame_time)

        print(peaks)
        print(phase)
        for pk in peaks:
            freq = freqs[pk]
            mag = magnitude[pk]
            phi = phases[pk]
            frame_signal += mag * np.cos(2 * np.pi * freq * frame_time + phi)

        # Overlap-Add
        start = i * hop_length
        end = start + win_length
        reconstructed_signal[start:end] += frame_signal
        overlap_counter[start:end] += 1

    # Avoid division by zero
    overlap_counter[overlap_counter == 0] = 1
    reconstructed_signal /= overlap_counter  # Normalize overlapping areas

    # Plot original and reconstructed signals
    t = np.arange(len(signal)) / sampling_rate
    t_rec = np.arange(len(reconstructed_signal)) / sampling_rate

    plt.figure(figsize=(12, 5))
    plt.plot(t, signal, label='Original signal')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Manual STFT Reconstruction from Magnitude & Phase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(t_rec, reconstructed_signal, label='Reconstructed signal (manual)', linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Manual STFT Reconstruction from Magnitude & Phase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    freqs, amplitudes = fft_analysis(reconstructed_signal, sampling_rate)

    plt.figure(figsize=(12, 5))
    plt.plot(freqs, amplitudes, label='Reconstructed signal (manual)', linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Manual STFT Reconstruction from Magnitude & Phase")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


        # peaks, peak_vals = detect_peaks(magnitude)
        # print(f"Time frame {i}: Peaks at bins {peaks} with magnitudes {peak_vals}")


    print(t)
    print(len(t))


    # x, v = fourier_analytical(signal)

    pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # save_numpy_file(output_path=output_path.with_suffix(".v.npy"), data=v)
    # save_numpy_file(output_path=output_path.with_suffix(".x.npy"), data=x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Signal magnetic_force-membrane_oscillation converter for loudspeaker signals.
        Analytical solution for STFT decomposition.
        """
    )
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "--input_signal_path",
        type=str,
        default="examples/cosine_wave.npy",
        help="Input signal representing external lorentz force.",
    )
    parser.add_argument(
        "--sampling_rate",
        type=int,
        default=48000,
        help="Input signal sampling rate for time domain.",
    )
    parser.add_argument(
        "--loudspeaker_params",
        type=str,
        default="examples/prv_audio_6MB400_8ohm.json",
        help="File representing loudspeaker parameters.",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=2048,
        help="Number of FFT points. Defines the frequency resolution of the STFT.",
    )
    parser.add_argument(
        "--win_length",
        type=int,
        default=2048,
        help="Length of the window applied to each segment for STFT. Should be <= n_fft.",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=512,
        help="Number of samples between successive STFT windows.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/x.npy",
        help="Output path.",
    )

    args = parser.parse_args()

    main(
        input_signal_path=Path(args.input_signal_path),
        loudspeaker_params=Loudspeaker.from_json(Path(args.loudspeaker_params)),
        output_path=Path(args.output_path),
        sampling_rate=args.sampling_rate,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
    )