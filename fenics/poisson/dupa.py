import os

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

# Mesh generation: Create a 2D square mesh for the diaphragm
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 50, 50)

# Define function spaces
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # Lagrange elements of order 1

# Material properties
density = 1.2  # Density of the diaphragm material (kg/m^2)
youngs_modulus = 1e6  # Young's modulus (Pa)
poisson_ratio = 0.3  # Poisson's ratio
thickness = 0.01  # Thickness of the diaphragm (m)

# Assemble the mass matrix
m = dolfinx.fem.locate_dofs_geometrical(V)
mass_matrix = dolfinx.fem.assemble_matrix(m, mesh)

# Assemble the stiffness matrix (plane stress)
E = youngs_modulus
nu = poisson_ratio
mu = E / (2 * (1 + nu))
lambda_ = (E * nu) / ((1 + nu) * (1 - 2 * nu))


# Define the stiffness matrix based on plane stress theory
def stiffness_matrix(u, v):
    return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx


stiffness_matrix = dolfinx.fem.assemble_matrix(stiffness_matrix, mesh)

# Damping term (simplified linear damping)
damping_coeff = 1.0
damping_matrix = damping_coeff * mass_matrix

# Define the force term (simple harmonic excitation)
force_frequency = 50  # Frequency of the applied electromagnetic force (Hz)
force_amplitude = 1.0  # Amplitude of the applied force

# Apply a simple harmonic force on the diaphragm
force = dolfinx.fem.Function(V)
x = force.x
force.vector[:] = force_amplitude * np.sin(2 * np.pi * force_frequency * x)

# Set initial conditions
u0 = dolfinx.fem.Function(V)
v0 = dolfinx.fem.Function(V)

# Define output directory
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define time-stepping solver for the system of equations
time_steps = 1000
dt = 0.01  # Time step size
t = 0  # Start time

# Time-stepping loop
for step in range(time_steps):
    # Displacement, velocity, and acceleration
    u = u0 + v0 * dt
    v = v0 + force * dt / mass_matrix

    # Compute the equation of motion: M * d^2u/dt^2 + K * u = F
    rhs = mass_matrix * v + stiffness_matrix * u
    rhs -= force

    # Solve the linear system
    petsc_solver = PETSc.KSP().create(MPI.COMM_WORLD)
    petsc_solver.setOperators(mass_matrix)
    petsc_solver.solve(rhs, u)

    # Update the displacement and velocity
    u0[:] = u[:]
    v0[:] = v[:]

    # Increment time
    t += dt

    # Save the displacement figure every few steps
    if step % 100 == 0:
        plt.clf()
        plt.imshow(u.x, cmap="inferno")
        plt.title(f"Displacement at time = {t:.2f}s")
        plt.colorbar()

        # Save the figure to the output directory
        figure_path = os.path.join(output_dir, f"displacement_{step:04d}.png")
        plt.savefig(figure_path)
        plt.close()

# Print completion message
print(f"Simulation complete. Figures saved to {output_dir}")
