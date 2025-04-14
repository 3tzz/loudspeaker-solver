import os
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import petsc4py
import petsc4py.PETSc as PETSc
import ufl
from dolfinx import geometry
from dolfinx.fem import Constant, FiniteElement, Function, FunctionSpace, functionspace
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.typing import Scalar
from ufl import Measure, ds, dx, grad, inner

script_path = Path(__file__).resolve()  # Full path of the script
script_dir = script_path.parent  # Directory containing the script

# name = "air_volume.msh"
# path = Path(script_dir, name)
output_path = Path("./fenics/first/output/")
output_path.mkdir(exist_ok=True, parents=True)

mesh_path = Path("mesh/air_volume/air_volume.msh")

# import the gmsh generated mesh
msh, coll, facet_tags, *_ = gmshio.read_from_msh(
    str(mesh_path), MPI.COMM_WORLD, 0, gdim=3
)

# Define frequency range domain
f_axis = np.arange(50, 1000, 5)
# print(f_axis)

V = functionspace(msh, ("Lagrange", 1))

# Define trial and test function
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


# Microphone location
mic = np.array([0.05, 0.06, 0.2])

# Defining air characteristics
c0 = 340  # sound speed [m/s]
rho_0 = 1.225  # air density [kg/m^3)]

omega = Constant(msh, PETSc.ScalarType(0))  # acoustic wave number [rad/m]
k0 = Constant(msh, PETSc.ScalarType(0))  # acoustic wave number [rad/m]


# Normal velocity boundary condition
v_n = 0.01  # NOTE: parameter to fine tune and select

# Surface impedance
Z_s = rho_0 * c0

# Defining bilinear and linear forms
ds = Measure("ds", domain=msh, subdomain_data=facet_tags)

# Weak form
a = (
    inner(grad(u), grad(v)) * dx - k0**2 * inner(u, v) * dx
)  # bilinear (liniowe względem jednego i drugiego w zależności od ) form

L = inner(1j * omega * rho_0 * v_n, v) * ds(
    2
)  # Velocity boundary condition week form, ds asseble to surface 2

uh = Function(V)
uh.name = "pressure"

solver = LinearProblem(
    a,
    L,
    u=uh,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)


def frequency_loop(nf):
    val = f_axis[nf]
    k0.value = 2 * np.pi * val / c0
    omega.value = 2 * np.pi * val
    solver.solve()

    if val % 100 == 0:
        with XDMFFile(
            msh.comm,
            str(output_path.joinpath("air_solution" + str(val) + ".xdmf")),
            "w",
        ) as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(uh)

    # Microphone point evaluation (Ensure correct shape and dtype)
    points = np.array([mic], dtype=np.float64)

    # Bounding box tree for point location
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)

    # Compute collisions
    cell_candidates = geometry.compute_collisions_points(
        bb_tree, points
    )  # Shape should be (N, 3)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)

    cells = []
    points_on_proc = []

    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)

    # Evaluate solution at microphone point
    if len(cells) > 0:
        p_values = uh.eval(points_on_proc, cells)
    else:
        p_values = np.array([0.0])  # Default value if point is outside the domain

    return p_values


if __name__ == "__main__":

    nf = range(0, len(f_axis))
    print("Computing...")
    pool = Pool(4)
    p_mic = pool.map(frequency_loop, nf)
    pool.close()
    pool.join()

    print("Saving results...")
    Re_p = np.real(p_mic)
    Im_p = np.imag(p_mic)

    stack_array = np.column_stack((f_axis, Re_p, Im_p))
    np.savetxt(
        str(output_path.joinpath("p_mic_spectrum.csv")), stack_array, delimiter=","
    )

    fig = plt.figure()
    plt.plot(f_axis, 20 * np.log10(np.abs(np.array(p_mic) / 2e-5)))
    plt.savefig(
        str(output_path.joinpath("frequency_response.png")), dpi=300
    )  # Save as PNG with high resolution
    plt.close(fig)  # Close the figure to free memory
