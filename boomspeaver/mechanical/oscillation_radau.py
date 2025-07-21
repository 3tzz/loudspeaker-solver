# pylint: disable=missing-module-docstring
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.data import load_numpy_file, save_numpy_file
from boomspeaver.tools.signal.make_sound import generate_time_domain


def radau_mass_spring_damper(
    t: np.ndarray, force: np.ndarray, m: float, c: float, k: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates the mass-spring-damper system using the implicit Radau method.

    Parameters:
    - t: Time array
    - force: External force array (F(t)) at each time step
    - m: Mass
    - c: Damping
    - k: Spring stiffness

    Returns:
    - x(t): Displacement array
    - v(t): Velocity array
    """

    # Interpolate force for continuous evaluation
    force_interp = interp1d(t, force, kind="linear", bounds_error=False, fill_value=0.0)

    def ode_system(ti, y):
        x, v = y
        F_t = force_interp(ti)
        dxdt = v
        dvdt = (F_t - c * v - k * x) / m
        return [dxdt, dvdt]

    y0 = [0.0, 0.0]
    sol = solve_ivp(
        ode_system,
        (t[0], t[-1]),
        y0,
        t_eval=t,
        method="Radau",
        vectorized=False,
        rtol=1e-6,
        atol=1e-9,
    )

    return sol.y[0], sol.y[1]

def calculate_mechanical_parameters(
    fr: float, m: float, q: float
) -> tuple[float, float]:
    """
    Calculate mechanical parameters for the loudspeaker:
    - Stiffness (k)
    - Damping factor (c)

    :param fr: Resonance frequency in Hz
    :param m: Moving mass in kg
    :param q: Quality factor
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

    t = generate_time_domain(sampling_rate=sampling_rate, n_samples=len(signal))

    fr = loudspeaker_params.thiele_small.fS
    m = loudspeaker_params.moving_mass.MMS
    q = loudspeaker_params.thiele_small.quality_factors.QTS

    k, c = calculate_mechanical_parameters(fr, m, q)

    x_signal, v_signal = radau_mass_spring_damper(
        t=t, force=signal, m=m, c=c, k=k
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
        description="""
        Signal magnetic_force-membrane_oscillation converter for loudspeaker signals.
        Using the implicit Radau method.
        """
    )
    parser.add_argument(
        "--input_signal_path",
        type=str,
        default="examples/cosine_wave.npy",
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
        default="output/radau.npy",
        help="Output path.",
    )

    args = parser.parse_args()

    main(
        input_signal_path=Path(args.input_signal_path),
        loudspeaker_params=Loudspeaker.from_json(Path(args.loudspeaker_params)),
        output_path=Path(args.output_path),
    )