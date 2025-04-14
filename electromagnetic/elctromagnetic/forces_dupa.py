import matplotlib.pyplot as plt
import numpy as np

# Parametry
m = 0.0098  # Masa w kg
f = 440  # Częstotliwość w Hz
dt = 1e-5  # Krok czasowy
T = 0.01  # Czas symulacji
n_steps = int(T / dt)

# Przygotowanie tablic
t_values = np.linspace(0, T, n_steps)
F_values = 5 * np.sin(2 * np.pi * f * t_values)  # Siła
v_values = np.zeros(n_steps)  # Prędkość, startujemy od zera

# Pętla symulacyjna (Euler symplektyczny)
for i in range(n_steps - 1):
    a = F_values[i] / m  # Przyspieszenie
    v_values[i + 1] = v_values[i] + a * dt  # Aktualizacja prędkości

# Wykres
plt.figure(figsize=(8, 5))
plt.plot(t_values, v_values, label="Prędkość v(t)", color="b")
plt.xlabel("Czas [s]")
plt.ylabel("Prędkość [m/s]")
plt.title("Prędkość masy w czasie")
plt.legend()
plt.grid()
plt.show()
