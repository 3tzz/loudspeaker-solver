from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def create_random_vector(
    length: int,
    seed: int = None,
    rng: np.random.Generator = None,
    value_range: tuple[float, float] | None = None
) -> np.ndarray:
    """
    Create a random vector of given length with optional scaling.
    """
    if seed is not None:
        np.random.seed(seed)
    if rng is None:
        y = np.random.randn(length)
    else:
        y = rng.standard_normal(length)

    if value_range:
        y = scale_to_range(y, value_range)
    return y

def scale_to_range(y: np.ndarray, value_range: tuple[float, float] = (0, 1)) -> np.ndarray:
    """
    Scale the input vector y to the specified range [a, b].
    """
    assert isinstance(y, np.ndarray)
    assert isinstance(value_range, tuple)
    assert len(value_range) == 2
    a, b = value_range
    y_min, y_max = y.min(), y.max()
    if y_max == y_min:
        return np.full_like(y, a)
    scaled = a + (b - a) * (y - y_min) / (y_max - y_min)
    return scaled

def shift_vector(vector: np.ndarray, to_idx: int, to_val: float) -> np.ndarray:
    """
    Shift vector to concrete value.
    """
    assert isinstance(vector, np.ndarray)
    assert isinstance(to_idx, int)
    assert isinstance(to_val, float)
    assert -len(vector) <= to_idx < len(vector)
    diff=to_val - vector[to_idx]
    return vector + diff

def mirror_vector(vector: np.ndarray, duplicate_center: bool = False) -> np.ndarray:
    """
    Mirror vector to make symmetric.
    """
    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1

    if duplicate_center:
        mirrored = np.concatenate([vector[::-1], vector])
    else:
        mirrored = np.concatenate([vector[1:][::-1], vector])
    return mirrored

def spline_interpolate(values: np.ndarray) -> CubicSpline:
    """
    Interpolate values between values using Cubic Spline interpolator for domain 0 < x < 1 (easy to scale).
    """
    x = np.linspace(0, 1, len(values))
    return CubicSpline(x, values)

if __name__ == "__main__":

    v=create_random_vector(5)
    print(f"Random points: {v}")

    v_scaled=scale_to_range(v, value_range=(-10.0,10.0))
    print(f"Scaled points: {v_scaled}")

    v_shifted=shift_vector(v_scaled, to_idx=-1, to_val=0.0)
    print(f"Shifted points: {v_shifted}")

    v_mirrored=mirror_vector(v_shifted)
    print(f"Mirrored points: {v_mirrored}")

    domain = np.linspace(0, 1, 200)
    spline=spline_interpolate(v_mirrored)

    interpolated_values=spline(domain)

    plt.plot(np.linspace(0,1,len(v_mirrored)),v_mirrored, 'o', label='Data points')
    plt.plot(domain, interpolated_values, '-', label='Cubic spline')
    plt.legend()
    plt.title("Cubic spline interpolation of symmetric random points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Membrane
    R = 1.0
    res = 500
    x = np.linspace(-R, R, res)
    y = np.linspace(-R, R, res)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    r_norm = np.clip(r, 0, 1)
    Z = spline(r_norm)

    Z[r > 1] = np.nan

    plt.figure(figsize=(6, 5))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.colorbar(label='Height')
    plt.title('Radially Symmetric Membrane Shape from Cubic Spline')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()

    plt.show()