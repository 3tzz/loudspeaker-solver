import dolfinx
import mpi4py.MPI  # Import MPI module here
import numpy as np
from dolfinx import default_scalar_type
from petsc4py import PETSc
from ufl import *

# Create mesh (unit square for simplicity)
mesh = dolfinx.mesh.create_unit_square(
    mpi4py.MPI.COMM_WORLD, 20, 20
)  # Using mpi4py.MPI.COMM_WORLD

# Define function space for displacement field (scalar displacement)
V = dolfinx.fem.functionspace(mesh, ("CG", 1))

# Trial and test functions
u = dolfinx.fem.Function(V)  # Displacement function
v = TestFunction(V)  # Test function


m = Constant(mesh, PETSc.ScalarType(0))

# Weak form for the displacement problem (simplified for geometry only)
a = inner(grad(u), grad(v)) * dx  # Gradient of displacement (basic Laplacian)
L = Constant(mesh, PETSc.ScalarType(0).real) * v * dx

# Apply boundary condition (fix displacement on boundary)
u_bc = dolfinx.fem.Function(V)
u_bc.vector.set(0.0)  # Fixed boundary condition (no displacement on boundary)
bc = dolfinx.fem.DirichletBC(
    u_bc, dolfinx.fem.locate_dofs_geometrical(mesh, lambda x: np.full_like(x[0], True))
)

# Create the problem and solve it
problem = dolfinx.fem.petsc.LinearProblem(
    a, L, bcs=[bc], petsc_options={"ksp_type": "cg"}
)
u = problem.solve()

# Post-processing: Visualize the displacement field
import matplotlib.pyplot as plt
import pyvista

# Extract the solution and convert to coordinates
coords = mesh.geometry.x
displacement = u.vector.get_local()

# Plotting the displacement field (in 2D)
mesh_points = coords[:, :2]  # Use the first two coordinates for 2D plot
displacement_field = displacement.reshape(
    (mesh_points.shape[0], 2)
)  # Assuming 2D displacement

# Create a PyVista plot
plotter = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(mesh_points, displacement_field)
plotter.add_mesh(grid, scalars=displacement_field[:, 0], cmap="coolwarm")
plotter.show()
