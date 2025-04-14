from pathlib import Path

import matplotlib as mpl
import numpy as np
import pyvista
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    set_bc,
)
from mpi4py import MPI
from petsc4py import PETSc

from electromagnetic.acoustic.tutorial.test_tutorial import initial_condition


def initial_condition(x):
    return np.zeros_like(x[0])


def FEM_wave_equation(domain, T, N, xs, output_directory, neumann_bc=True, c=343):
    assert isinstance(output_directory, Path)
    output_directory.mkdir(parents=True, exist_ok=False)
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Define previous and second-last solutions
    u1 = fem.Function(V)
    u0 = fem.Function(V)
    u1.interpolate(initial_condition)  # Initial condition
    u0.interpolate(initial_condition)  # Initial condition

    # Define boundary conditions
    if neumann_bc:
        bc = None
    else:
        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        bc = fem.dirichletbc(
            PETSc.ScalarType(0),
            fem.locate_dofs_topological(V, fdim, boundary_facets),
            V,
        )

    xdmf = io.XDMFFile(
        domain.comm,
        str(Path(output_directory, "wave_equation.xdmf")),
        "w",
    )
    xdmf.write_mesh(domain)

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Define bilinear and linear forms
    a = (
        ufl.inner(u, v) * ufl.dx
        + fem.Constant(domain, PETSc.ScalarType(T**2 * c**2))
        * ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v))
        * ufl.dx
    )
    L = 2 * u1 * v * ufl.dx - u0 * v * ufl.dx

    # Compute solution for all time-steps
    u = fem.Function(V)

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    A = assemble_matrix(bilinear_form, bcs=[bc])
    # A = assemble_matrix(bilinear_form)
    A.assemble()
    b = create_vector(linear_form)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    xdmf.write_function(u, 0.0)
    t = 0
    for n in range(N):
        t += T
        with b.localForm() as loc:
            loc.set(0)
        assemble_vector(b, linear_form)
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        # Point source at t=0
        if n == 0:
            # Add a smooth Gaussian pulse manually
            f = fem.Function(V)
            f.interpolate(
                lambda x: np.exp(-1e3 * ((x[0] - xs[0]) ** 2 + (x[1] - xs[1]) ** 2))
            )
            f_vec = create_vector(fem.form(f * v * ufl.dx))
            assemble_vector(f_vec, fem.form(f * v * ufl.dx))
            f_vec.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )
            b.axpy(1.0, f_vec)

        solver.solve(b, u.x.petsc_vec)
        u.x.scatter_forward()

        u0.x.array[:] = u1.x.array
        u1.x.array[:] = u.x.array
        xdmf.write_function(u, t)
    xdmf.close()
    return u


if __name__ == "__main__":
    # Define simulation parameters
    T = 1.0  # Time step
    N = 100  # Number of steps
    xs = np.array([0.5, 0.5], dtype=np.float64)
    output_dir = Path("./electromagnetic/acoustic/tutorial/output/")

    # Set up mesh
    nx, ny = 50, 50
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-2, -2]), np.array([2, 2])],
        [nx, ny],
        mesh.CellType.triangle,
    )

    # Run the simulation
    u = FEM_wave_equation(domain, T, N, xs, output_dir, neumann_bc=False)
    # u = FEM_wave_equation(domain, T, N, xs, output_dir)
