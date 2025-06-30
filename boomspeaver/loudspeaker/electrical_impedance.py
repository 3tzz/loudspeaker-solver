# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring


from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from scipy.interpolate import interp1d

from boomspeaver.tools.data import get_repo_dir, load_csv_file


@dataclass
class ImpedanceData:
    frequencies: np.ndarray  # Hz
    impedances: np.ndarray  # Ohms

    @classmethod
    def from_csv(
        cls,
        input_path: Union[str, Path],
        freq_col: str = "frequency",
        imp_col: str = "impedance",
    ) -> "ImpedanceData":
        """Load impedance data using helper method."""
        input_path = Path(input_path)
        df = load_csv_file(input_path)
        df.columns = [freq_col, imp_col]

        freqs = df[freq_col].astype(float).to_numpy()
        imps = df[imp_col].astype(float).values

        sorted_indices = np.argsort(freqs)
        freqs_sorted = freqs[sorted_indices]
        imps_sorted = imps[sorted_indices]

        return cls(frequencies=freqs_sorted, impedances=imps_sorted)

    def extrapolate(self, min_freq: int = 0, max_freq: int = 24000):
        """Extrapolate frequency and impedance values."""
        min_params_freq = self.frequencies[0]
        max_params_freq = self.frequencies[-1]

        interp_fn = interp1d(
            self.frequencies,
            self.impedances,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )

        new_freqs = []
        new_imps = []

        if min_freq < min_params_freq:
            delta = self.frequencies[1] - self.frequencies[0]
            down_freqs = np.arange(min_freq, min_params_freq, delta)
            down_values = interp_fn(down_freqs)
            new_freqs.extend(down_freqs)
            new_imps.extend(down_values)

        new_freqs.extend(self.frequencies)
        new_imps.extend(self.impedances)

        if max_freq > max_params_freq:
            delta = self.frequencies[-1] - self.frequencies[-2]
            up_freqs = np.arange(max_params_freq + delta, max_freq + delta, delta)
            up_values = interp_fn(up_freqs)
            new_freqs.extend(up_freqs)
            new_imps.extend(up_values)

        sorted_idx = np.argsort(new_freqs)
        self.frequencies = np.array(new_freqs)[sorted_idx]
        self.impedances = np.array(new_imps)[sorted_idx]

    def interpolate_impedance(self, f: float) -> complex:
        """Interpolate impedance at frequency `f` using linear interpolation."""
        if f < self.frequencies.min() or f > self.frequencies.max():
            raise ValueError(f"Frequency {f} Hz is outside the data range.")

        return float(np.interp(f, self.frequencies, self.impedances))

    def get_impedance(
        self, f: float, max_error: float = 1, interpolate: bool = True
    ) -> complex:
        """Get nearest impedance value according to frequency if smaller than max error."""
        idx = np.abs(self.frequencies - f).argmin()
        nearest_freq = self.frequencies[idx]
        error = abs(nearest_freq - f)

        if max_error is not None and error > max_error:
            if interpolate:
                return self.interpolate_impedance(f)
            else:
                raise ValueError(
                    f"No frequency close to {f} Hz within {max_error} Hz (nearest is {nearest_freq} Hz)"
                )
        return self.impedances[idx]


if __name__ == "__main__":

    repo_dir = get_repo_dir(run_type="python")
    # repo_dir = get_repo_dir(run_type="docker")
    input_config_path = repo_dir / "examples/electrical_impedance.csv"
    impedance = ImpedanceData.from_csv(input_path=input_config_path)
    print(impedance.get_impedance(420))
