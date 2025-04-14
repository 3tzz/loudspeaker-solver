"""
Solves the 2D Poisson Equation with homogeneous Dirichlet Boundary Conditions and a
constant forcing right hand side using the Finite Element Method with FEniCS in
Python.

    − ∇²u = f

u  : Vertical Displacement of a membrane
f  : Forcing right hand side
∇² : Laplace Operator

-----

Problem Setup:

A 2D square membrane is fixed on all four edges. A constant force is applied
over the entire domain.

                        u fixed to 0

                   +----------------------+
                  /↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ /
                 /↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ /
 u fixed to 0   /↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ /   u fixed to 0
               /↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ /
              /↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ /
             /↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ /
            +----------------------+
        
                u fixed to 0

The solution to the Poisson Equation is the deformation of the
membrane u, which depends on both spatial axes, x_0 & x_1.

-----

Weak Form:

    <∇u, ∇v> = <f, v>

<⋅, ⋅> indicates the inner product, which for functions
refers of a contraction to scalar and integration over
the domain.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.plot import vtk_mesh
from mpi4py import MPI

os.environ["DISPLAY"] = ""

N_POINTS_P_AXIS = 12
FORCING_MAGNITUDE = 1.0


def on_boundary(x):
    return (
        np.isclose(x[0], 0)  # Left edge
        | np.isclose(x[0], 1)  # Right edge
        | np.isclose(x[1], 0)  # Bottom edge
        | np.isclose(x[1], 1)  # Top edge
    )


domain = mesh.create_unit_square(MPI.COMM_WORLD, N_POINTS_P_AXIS, N_POINTS_P_AXIS)

x = ufl.SpatialCoordinate(domain)

V = fem.functionspace(domain, ("Lagrange", 1))

boundary_dofs = fem.locate_dofs_geometrical(V, on_boundary)
bc = fem.dirichletbc(default_scalar_type(0), boundary_dofs, V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

f = fem.Constant(domain, -FORCING_MAGNITUDE)

# Correct the form for real-valued Poisson equation
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx  # real-valued inner product
L = f * v * ufl.dx

problem = LinearProblem(
    a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)

uh = problem.solve()

print("Computed!")

# Visualize
script_path = Path(__file__).resolve()  # Full path of the script
script_dir = script_path.parent  # Directory containing the script

output_dir = script_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

# Define the path for the XDMF file
output_path = output_dir / "poisson_final.xdmf"

# Write mesh and solution to XDMF (only once)
with XDMFFile(domain.comm, str(output_path), "w") as xdmf:
    xdmf.write_mesh(domain)  # Write the mesh
    xdmf.write_function(uh)  # Write the solution

print(f"Solution saved to {output_path}")

# # Matplotlib
# # Get the solution data
# solution = uh.x.array
#
# # Reshape the solution to fit the mesh grid (assuming it’s a structured grid)
# x = domain.geometry.x[:, 0]
# y = domain.geometry.x[:, 1]
# X, Y = np.meshgrid(np.unique(x), np.unique(y))
# Z = solution.reshape(len(np.unique(y)), len(np.unique(x)))
#
# # Plot the solution
# plt.figure(figsize=(8, 6))
# cp = plt.contourf(X, Y, Z, 20, cmap="viridis")
# plt.colorbar(cp)
# plt.title("Solution to the Poisson Equation")
# plt.xlabel("x")
# plt.ylabel("y")
#
# # Save the plot to the output directory
# output_path = output_dir / "solution.png"
# plt.savefig(str(output_path))
# plt.close()  # Close the plot to free up resources
# print(f"Figure saved at {output_path}")

# # Create PyVista plotter
# plotter = pyvista.Plotter(
#     off_screen=True, screenshot_scale=1
# )  # Off-screen rendering for saving
# topology, cell_types, geometry = vtk_mesh(V)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# grid.point_data["u"] = uh.x.array
# plotter.add_mesh(grid, show_edges=True, scalars="u", cmap="viridis")
#
# # Save the figure
# output_path = output_dir / "solution.png"
# plotter.screenshot(output_path, scale=1)
#
# raise
