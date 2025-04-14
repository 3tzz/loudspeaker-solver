import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from sympy import integrate, lambdify, sin, symbols

# Define symbols
t = symbols("t")
t_values = np.linspace(0, 10, 5000)  # Time from 0 to 10 seconds
m = 0.0098  # Mass in kg
f = 440
a = 5

F_L = a * sin(2 * t)


def F_func(t):
    return a * np.sin(2 * t)


# Define system of ODEs
def model(y, t):
    x, v = y  # Unpack position and velocity
    dxdt = v  # First derivative (velocity)
    dvdt = F_func(t) / m  # Second derivative (acceleration)
    return [dxdt, dvdt]


# Initial conditions [x0, v0]
y0 = [0, 0]

# Solve ODE numerically
solution = odeint(model, y0, t_values)
x_numerical = solution[:, 0]  # Extract displacement

# Compute velocity and displacement
v = integrate(F_L / m, t)
x = integrate(v, t)

# Convert symbolic expressions to numerical functions
F_func = lambdify(t, F_L, "numpy")  # Force function
x_func = lambdify(t, x, "numpy")  # Displacement function

# Evaluate force and displacement at each time point
F_values = F_func(t_values)
x_values = x_func(t_values)

# Plot the force and displacement
plt.figure(figsize=(8, 5))

plt.subplot(3, 1, 1)
plt.plot(t_values, F_values, label=r"$F_L(t)$ (Force)", color="r", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Magnitude")
plt.title("Force over Time")
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t_values, x_values, label=r"$x(t)$ (displacement)", color="b")
plt.xlabel("time (s)")
plt.ylabel("magnitude")
plt.title("Displacement over time")
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t_values, x_numerical, label=r"$x(t)$ (displacement)", color="b")
plt.xlabel("time (s)")
plt.ylabel("magnitude")
plt.title("Displacement numerical over time")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
