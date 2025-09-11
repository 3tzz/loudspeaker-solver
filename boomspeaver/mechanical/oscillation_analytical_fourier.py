import argparse
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.mechanical.oscillation_euler import calculate_mechanical_parameters
from boomspeaver.tools.data import load_numpy_file, save_numpy_file
from boomspeaver.tools.dsp.dsp import detect_peaks, fft_analysis, stft
from boomspeaver.tools.dsp.make_sound import generate_chord, generate_time_domain
from boomspeaver.tools.plot.configs import Axis, Line, Plotter, Points, Subplot
from boomspeaver.tools.plot.multi_plotter import MultiPlotter


class Solver:
    def solve(
        self, time: np.ndarray, initial_displacement: float, initial_velocity: float
    ):
        raise NotImplementedError


class HomogeneousOscillationSolver:
    def __init__(self, system: "HarmonicOscillatorParams"):
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

    def solve(
        self, time: np.ndarray, initial_displacement: float, initial_velocity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.solver.solve(time, initial_displacement, initial_velocity)


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

    def __init__(self, system: "HarmonicOscillatorParams"):
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

    def calculate_initial_constants(
        self, initial_displacement: float, initial_velocity
    ) -> tuple[float, float]:
        """
        Calculate A₁ and A₂:
            x(0) = A₁ + A₂ = x₀
            ẋ(0) = −A₁ ⋅ ω₊ − A₂ ⋅ ω₋ = v₀
        """
        A = np.array([[1, 1], [-self.omega_plus, -self.omega_minus]])
        b = np.array([initial_displacement, initial_velocity])
        A1, A2 = np.linalg.solve(A, b)
        return A1, A2

    def calculate_constant_part(
        self, time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The time dependent constants:
            e^(−ω₊ ⋅ t)
            e^(−ω₋ ⋅ t))
        """
        e1 = np.exp(-self.omega_plus * time)
        e2 = np.exp(-self.omega_minus * time)
        return e1, e2

    def solve(
        self, time: np.ndarray, initial_displacement: float, initial_velocity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The calculate response:
            x(t) = A₁ ⋅ e^(−ω₊ ⋅ t) + A₂ ⋅ e^(−ω₋ ⋅ t)
            ẋ(t) = −A₁ ⋅ ω₊ ⋅ e^(−ω₊ ⋅ t) − A₂ ⋅ ω₋ ⋅ e^(−ω₋ ⋅ t)
        """
        A1, A2 = self.calculate_initial_constants(
            initial_displacement=initial_displacement, initial_velocity=initial_velocity
        )

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

    def __init__(self, system: "HarmonicOscillatorParams"):
        self.system = system
        assert self.system.damping_type.lower() == "critical"

    def calculate_initial_constants(
        self, initial_displacement: float, initial_velocity: float
    ) -> tuple[float, float]:
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

    def solve(
        self, time: np.ndarray, initial_displacement: float, initial_velocity: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the response:
            x(t) = (A + B⋅t) ⋅ e^(−Γ/2 ⋅ t)
            ẋ(t) = [B − (Γ/2)⋅(A + B⋅t)] ⋅ e^(−Γ/2 ⋅ t)
        """
        A1, A2 = self.calculate_initial_constants(
            initial_displacement, initial_velocity
        )
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

    def __init__(self, system: "HarmonicOscillatorParams"):
        self.system = system
        assert self.system.damping_type.lower() == "underdamped"

    def calculate_initial_constants(
        self, initial_displacement: float, initial_velocity: float
    ) -> tuple[float, float]:
        """
        Solve for A and β using:
            x(0) = A ⋅ cos(β)
            ẋ(0) = −A ⋅ [ω₁ ⋅ sin(β) + Γ/2 ⋅ cos(β)]
        """
        Γ2 = self.system.total_c_by_2
        ω1 = self.system.omega1

        β = np.arctan2(
            ω1 * initial_displacement, Γ2 * initial_displacement + initial_velocity
        )

        A = initial_displacement / np.cos(β)

        return A, β

    def calculate_constant_part(self, time: np.ndarray) -> np.ndarray:
        """
        Compute e^(−Γ/2 ⋅ t)
        """
        return np.exp(-self.system.total_c_by_2 * time)

    def solve(
        self, time: np.ndarray, x0: float, v0: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return x(t), ẋ(t) for underdamped case
            x(t) = A ⋅ e^(−Γ/2 ⋅ t) ⋅ cos(ω₁ t − β)
            ẋ(t) = −A ⋅ e^(−Γ/2 ⋅ t) ⋅ [ω₁ ⋅ sin(ω₁ t − β) + Γ/2 ⋅ cos(ω₁ t − β)]
        """
        A, β = self.calculate_initial_constants(x0, v0)
        e1 = self.calculate_constant_part(time)

        phase = self.system.omega1 * time - β

        displacement = A * e1 * np.cos(phase)
        velocity = (
            -A
            * e1
            * (
                self.system.omega1 * np.sin(phase)
                + self.system.total_c_by_2 * np.cos(phase)
            )
        )

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

    def __init__(self, system: "HarmonicOscillatorParams"):
        self.system = system

    def calculate_amplitude(self, frequency: float, amplitude: float) -> float:
        """
        Calculate amplitude (A) for solution:
            x(t) = A ⋅ cos(ω t + δ - δ_res)
        The amplitude A is given by:
            A = (F₀ / m) / sqrt((ω₀² − ω²)² + (bω / m)²)
        """
        omega = (self.system.omega0**2 - frequency**2) ** 2
        amp = amplitude / self.system.m
        damp = ((self.system.total_c * frequency)) ** 2
        return amp / np.sqrt(omega + damp)

    def calculate_phase(self, frequency: np.ndarray) -> float:
        """
        Calculate phase (δ) for solution:
            x(t) = A ⋅ cos(ω t + δ - δ_res)
        The phase shift δ is given by:
            δ  = arctan( (b/m) ω / (ω₀² − ω²) )
        """
        numerator = (self.system.total_c) * frequency
        denominator = self.system.omega0**2 - frequency**2
        return np.arctan2(numerator, denominator)

    def solve(
        self, time: np.ndarray, frequency: float, amplitude: float, phase: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Driving force
            Fⱼ·cos(ωⱼ·t + φⱼ)
        Computes the particular (steady-state) solution of the driven damped harmonic oscillator:
            x(t) = A ⋅ cos(ω t + δ - δ_res)
            ẋ(t) = -A ⋅ ω ⋅ sin(ω t + δ - δ_res)
        """

        amplitude = self.calculate_amplitude(frequency, amplitude)
        angle = self.calculate_phase(frequency)

        span = frequency * time + angle - phase
        # span = frequency*time - phase
        # span = frequency*time

        displacement = amplitude * np.cos(span)
        velocity = -amplitude * frequency * np.sin(span)

        return displacement, velocity


class HarmonicOscillationParams:
    def __init__(
        self, mass: float, damping_coefficient: float, stiffness: float
    ) -> None:
        assert mass != 0.0
        self.m = mass
        self.c = damping_coefficient
        self.k = stiffness

    @cached_property
    def omega0(self):
        return np.sqrt(self.k / self.m)  # natural frequency

    @cached_property
    def total_c(self):
        return self.c / self.m  # total damping parameter

    @cached_property
    def total_c_by_2(self):
        return self.total_c / 2

    @cached_property
    def omega1(self):
        return np.sqrt(
            self.omega0**2 - (self.total_c_by_2) ** 2
        )  # damped natural frequency

    @cached_property
    def damping_type(self):
        square_omega1 = self.omega1**2
        if square_omega1 > 0:
            return "underdamped"
        elif square_omega1 == 0:
            return "critically"
        elif square_omega1 < 0:
            return "overdamped"
        else:
            NotImplementedError

    @classmethod
    def from_loudspeaker_params_file(cls, loudspeaker_params: Loudspeaker):
        assert isinstance(loudspeaker_params, Loudspeaker)
        k, c = calculate_mechanical_parameters(
            loudspeaker_params.thiele_small.fS,
            loudspeaker_params.moving_mass.MMS,
            loudspeaker_params.thiele_small.quality_factors.QTS,
        )
        return cls(
            mass=loudspeaker_params.moving_mass.MMS, damping_coefficient=c, stiffness=k
        )


class HarmonicOscillationSystem:
    """
    Solve the equation of motion for a linear damped harmonic oscillator with multiple harmonic external forces:
    m·ẍ(t) + c·ẋ(t) + k·x(t) = Σⱼ Fⱼ·cos(ωⱼ·t + φⱼ)
    https://phys.libretexts.org/Bookshelves/Classical_Mechanics/Variational_Principles_in_Classical_Mechanics_(Cline)/03%3A_Linear_Oscillators/3.06%3A_Sinusoidally-driven_linearly-damped_linear_oscillator
    """

    def __init__(self, loudspeaker_params: Loudspeaker) -> None:
        self.parameters = HarmonicOscillationParams.from_loudspeaker_params_file(
            loudspeaker_params=loudspeaker_params
        )
        self.homogeneus = HomogeneousOscillationSolver(self.parameters)
        self.particular = ParticularFrequency(self.parameters)

    def solve_homogeneous_part(
        self,
        time: np.ndarray,
        initial_displacement: np.ndarray,
        initial_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.homogeneus.solve(
            time=time,
            initial_displacement=initial_displacement,
            initial_velocity=initial_velocity,
        )

    def solve_particular_part(
        self,
        time: np.ndarray,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        phases: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert len(frequencies) == len(amplitudes) == len(phases)
        displacement = np.zeros_like(time)
        velocity = np.zeros_like(time)
        for frequency, amplitude, phase in zip(frequencies, amplitudes, phases):
            x, v = self.particular.solve(
                time=time,
                frequency=2 * np.pi * frequency,
                amplitude=amplitude,
                phase=phase,
            )
            displacement += x
            velocity += v
        return displacement, velocity

    def solve(
        self,
        time: np.ndarray,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        phases: np.ndarray,
        initial_displacement: float = 0.0,
        initial_velocity: float = 0.0,
    ):
        homogeneous_solution_displacement, homogeneous_solution_velocity = (
            self.solve_homogeneous_part(
                time=time,
                initial_displacement=initial_displacement,
                initial_velocity=initial_velocity,
            )
        )
        particular_solution_displacement, particular_solution_velocity = (
            self.solve_particular_part(
                time=time, frequencies=frequencies, amplitudes=amplitudes, phases=phases
            )
        )
        displacement_solution = (
            homogeneous_solution_displacement + particular_solution_displacement
        )
        velocity_solution = homogeneous_solution_velocity + particular_solution_velocity
        return displacement_solution, velocity_solution


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
    amplitude_frequency_threshold=0.1,
    verbose: bool = True,
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

    assert len(freqs) == spec.shape[0]

    oscillation_solver = HarmonicOscillationSystem(loudspeaker_params)

    n_frames = spec.shape[1]
    # frame_time = np.arange(win_length) / sampling_rate
    frame_time = np.arange(hop_length) / sampling_rate

    reconstructed_length = hop_length * (n_frames - 1) + win_length
    reconstructed_displacement = np.zeros(reconstructed_length)
    reconstructed_velocity = np.zeros(reconstructed_length)
    overlap_counter = np.zeros(reconstructed_length)
    for i, (frequencies, amplitudes, phases) in enumerate(
        frame_generator(freqs, spec, threshold=amplitude_frequency_threshold)
    ):
        print(f"Frame {i} - peaks: {frequencies}, {phases}")

        # Overlap-Add
        start = i * hop_length
        # end = start + win_length
        end = start + hop_length

        # # For validation test
        # normalization = np.zeros_like(frame_time)
        # for freq,mag,phi in zip(frequencies, amplitudes, phases):
        #     reconstructed_displacement[start:end] += np.cos(2 * np.pi * freq * frame_time + phi)

        # reconstructed_velocity[start:end] = reconstructed_displacement[start:end]

        displacement, velocity = oscillation_solver.solve(
            time=frame_time,
            frequencies=frequencies,
            amplitudes=amplitudes,
            phases=phases,
            initial_displacement=reconstructed_displacement[start],
            initial_velocity=reconstructed_velocity[start],
        )

        reconstructed_displacement[start:end] += displacement
        reconstructed_velocity[start:end] += velocity

        overlap_counter[start:end] += 1
    overlap_counter[overlap_counter == 0] = 1
    reconstructed_displacement /= overlap_counter
    reconstructed_velocity /= overlap_counter

    reconstructed_displacement = reconstructed_displacement[: len(signal)]
    reconstructed_velocity = reconstructed_velocity[: len(signal)]
    t = np.arange(len(signal)) / sampling_rate
    t_rec = np.arange(len(reconstructed_displacement)) / sampling_rate

    if verbose:
        plt.figure(figsize=(12, 5))
        plt.plot(t, signal, label="Original signal")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Manual STFT Reconstruction from Magnitude & Phase")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.plot(
            t_rec,
            reconstructed_displacement,
            label="Reconstructed signal (manual)",
            linestyle="--",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.title("Manual STFT Reconstruction from Magnitude & Phase")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        freqs, amplitudes = fft_analysis(signal, sampling_rate)

        plt.figure(figsize=(12, 5))
        plt.plot(freqs, amplitudes, label="Orginal signal (manual)")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.title("Manual STFT Reconstruction from Magnitude & Phase")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        freqs, amplitudes = fft_analysis(reconstructed_displacement, sampling_rate)

        plt.figure(figsize=(12, 5))
        plt.plot(
            freqs, amplitudes, label="Reconstructed signal (manual)", linestyle="--"
        )
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.title("Manual STFT Reconstruction from Magnitude & Phase")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_numpy_file(
        output_path=output_path.with_suffix(".v.npy"), data=reconstructed_displacement
    )
    save_numpy_file(
        output_path=output_path.with_suffix(".x.npy"), data=reconstructed_velocity
    )


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
        default="examples/chord_signal.npy",
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
        default="output/cosine_wave_analytical.npy",
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

