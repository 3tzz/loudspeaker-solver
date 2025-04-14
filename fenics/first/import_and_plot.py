from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

output_path = Path("./fenics/first/output/")
output_path.mkdir(exist_ok=True, parents=True)

l1 = np.loadtxt(str(output_path / "p_mic_spectrum.csv"), delimiter=",")
l2 = np.loadtxt(str(output_path / "p_mic_spectrum_2.csv"), delimiter=",")

f_axis = l1[:, 0]
f_axis = l2[:, 0]

p1 = l1[:, 1] + 1j * l1[:, 2]
p2 = l2[:, 1] + 1j * l2[:, 2]

fig = plt.figure()
plt.plot(f_axis, 20 * np.log10(np.abs(p1) / 2e-5))
plt.plot(f_axis, 20 * np.log10(np.abs(p2) / 2e-5))

plt.savefig(
    str(output_path / "frequency_response_all.png"), dpi=300
)  # Save as PNG with high resolution
plt.close(fig)  # Close the figure to free memory
