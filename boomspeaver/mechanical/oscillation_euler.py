import argparse
from pathlib import Path

import numpy as np

from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.data import load_numpy_file, save_numpy_file
from boomspeaver.tools.dsp.make_sound import generate_time_domain


def euler_mass_spring_damper(
    t: np.ndarray, dt: float, force: np.ndarray, m: float, c: float, k: float
) -> tuple[np.ndarray, np.ndarray]:
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
    - force: External force array (F(t)) at each time step
    - m: Mass of the object (kg)
    - c: Damping coefficient (N·s/m)
    - k: Spring constant (N/m)

    Note:
    - This method works for any external force F(t) that is provided as an array
      where the force is defined at each time step.
    - The time step (dt) is assumed to be constant, so the method is appropriate for
      problems where the time resolution does not change over time.
    """
    assert len(t) == len(force)
    x = np.zeros(len(t))
    v = np.zeros(len(t))

    for n in range(1, len(t)):
        v[n] = (
            v[n - 1] + (force[n - 1] / m - (c / m) * v[n - 1] - (k / m) * x[n - 1]) * dt
        )
        x[n] = x[n - 1] + v[n - 1] * dt

    return x, v


def calculate_mechanical_parameters(
    fr: float, m: float, q: float
) -> tuple[float, float]:
    """
    Calculate mechanical parameters for the loudspeaker:
    - Stiffness (k)
    - Damping factor (c)

    :param fr: Resonance frequency in Hz
    :param m: Moving mass in kg
    :param q_m: Quality factor
    """

    # Stiffness
    k = (2 * np.pi * fr) ** 2 * m

    # Damping
    c = np.sqrt(m * k) / q

    return k, c


def diaphragm_motion(
    signal: np.ndarray,
    loudspeaker_params: Loudspeaker,
    sampling_rate: int = 48000,
) -> tuple[np.ndarray, np.ndarray]:
    assert isinstance(loudspeaker_params, Loudspeaker)
    assert isinstance(signal, np.ndarray)

    dt = 1 / sampling_rate
    t = generate_time_domain(sampling_rate=sampling_rate, n_samples=len(signal))

    fr = loudspeaker_params.thiele_small.fS
    m = loudspeaker_params.moving_mass.MMS
    q = loudspeaker_params.thiele_small.quality_factors.QTS

    k, c = calculate_mechanical_parameters(fr, m, q)

    x_signal, v_signal = euler_mass_spring_damper(
        t=t, dt=dt, force=signal, m=m, c=c, k=k
    )
    return x_signal, v_signal


def main(
    input_signal_path: Path,
    loudspeaker_params: Loudspeaker,
    output_path: Path,
    sampling_rate: int = 48000,
) -> None:

    assert isinstance(input_signal_path, Path)
    assert input_signal_path.exists()
    assert input_signal_path.suffix == ".npy"
    assert isinstance(output_path, Path)
    assert isinstance(loudspeaker_params, Loudspeaker)

    signal = load_numpy_file(input_signal_path)

    x, v = diaphragm_motion(
        signal=signal,
        loudspeaker_params=loudspeaker_params,
        sampling_rate=sampling_rate,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_numpy_file(output_path=output_path.with_suffix(".v.npy"), data=v)
    save_numpy_file(output_path=output_path.with_suffix(".x.npy"), data=x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal magnetic_force-membrane_oscillation converter for loudspeaker signals."
    )
    subparsers = parser.add_subparsers(dest="command")
    parser.add_argument(
        "--input_signal_path",
        type=str,
        default="examples/chord_signal.npy",
        help="Input current signal.",
    )
    parser.add_argument(
        "--loudspeaker_params",
        type=str,
        default="examples/prv_audio_6MB400_8ohm.json",
        help="File representing loudspeaker parameters.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/chord_signal_mechanical.npy",
        help="Output path.",
    )

    args = parser.parse_args()

    main(
        input_signal_path=Path(args.input_signal_path),
        loudspeaker_params=Loudspeaker.from_json(Path(args.loudspeaker_params)),
        output_path=Path(args.output_path),
    )

