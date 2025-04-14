import time

import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI
from petsc4py import PETSc

# Create a unit square mesh with 50 cells in each direction
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define mesh and function space
nx = 50  # Number of cells in x-direction
ny = 50  # Number of cells in y-direction
mesh2d = mesh.create_unit_square(comm, nx, ny)

# Define the function space for magnetic vector potential A
V = fem.functionspace(mesh2d, ("Lagrange", 1))

# Define the test function v
v = ufl.TestFunction(V)

# Define the current density (time-dependent)
frequency = 1000  # Frequency in Hz


def current_signal(t):
    return np.sin(2 * np.pi * frequency * t)


# Define time and parameters for the simulation
t = 0.0  # Start time
dt = 0.01  # Time step
T = 1.0  # Total time

# Define magnetic permeability of free space
mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability

# Define weak form for Maxwell's equations (Ampère's law)
A = fem.Function(V)  # Magnetic vector potential
J = fem.Function(V)  # Current density


# Time-dependent current density interpolation
def current_density_expr(x, t):
    # Time-dependent current density
    return current_signal(t)


# Create a time-stepping loop
for t in np.arange(0, T, dt):
    # Interpolate current density into the function space for each time step
    current_density = fem.Function(V)
    current_density.interpolate(lambda x: current_density_expr(x, t))

    # Weak form: div(B) = 0 and curl(B) = J
    F = (
        ufl.inner(ufl.curl(A), ufl.curl(v)) * ufl.dx
        - ufl.inner(current_density, v) * ufl.dx
    )

    # Apply boundary conditions (e.g., zero magnetic vector potential on the boundary)
    bc = fem.dirichletbc(
        np.zeros(1), fem.locate_dofs_geometrical(V, lambda x: np.full_like(x[0], True))
    )

    # Create the problem and solver
    problem = fem.petsc.LinearProblem(F, bcs=[bc])
    solver = problem.create_solver()

    # Solve for the magnetic field A
    solver.solve()

    # Extract the magnetic field B from A (B = curl(A) in 2D)
    B = fem.Function(V)
    B.interpolate(lambda x: np.cross(np.gradient(A.vector.array())))

    # Visualize the magnetic field using matplotlib
    x_coords = mesh2d.geometry.x
    y_coords = mesh2d.geometry.y
    B_values = B.vector.array

    plt.quiver(x_coords, y_coords, B_values)
    plt.title(f"Magnetic Field (B) at time {t:.2f}s")
    plt.show()

    # Optionally, break the loop if you don't need to visualize for every time step
    # break
