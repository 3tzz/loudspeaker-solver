from pathlib import Path

import dolfinx
from dolfinx.fem import functionspace
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI


def load_mesh(mesh_path: Path, mesh_dimension: int):
    assert isinstance(mesh_path, Path)
    assert mesh_path.exists()
    assert isinstance(mesh_dimension, int)
    domain, coll, facet_tags, *_ = gmshio.read_from_msh(
        str(mesh_path), MPI.COMM_WORLD, 0, gdim=mesh_dimension
    )
    return domain, coll, facet_tags


def FEM_wave_equation(mesh, T, N, xs, neumann_bc=True, c=343):

    # define function space
    V = dolfin.FunctionSpace(mesh, "CG", 1)

    # define previous and second-last solution
    u1 = dolfin.interpolate(dolfin.Constant(0.0), V)
    u0 = dolfin.interpolate(dolfin.Constant(0.0), V)

    # define boundary conditions
    if neumann_bc:
        bcs = None
    else:
        bcs = dolfin.DirichletBC(V, dolfin.Constant(0.0), "on_boundary")

    # define variational problem
    u = dolfin.TrialFunction(V)
    v = dolfin.TestFunction(V)

    a = (
        dolfin.inner(u, v) * dolfin.dx
        + dolfin.Constant(T**2 * c**2)
        * dolfin.inner(dolfin.nabla_grad(u), dolfin.nabla_grad(v))
        * dolfin.dx
    )
    L = 2 * u1 * v * dolfin.dx - u0 * v * dolfin.dx

    # compute solution for all time-steps
    u = dolfin.Function(V)

    for n in range(N):
        A, b = dolfin.assemble_system(a, L, bcs)
        # define inhomogenity
        if n == 0:
            delta = dolfin.PointSource(V, xs, 1)
            delta.apply(b)
        # solve variational problem
        dolfin.solve(A, u.vector(), b)
        u0.assign(u1)
        u1.assign(u)

    return u


def fem_wave_equation(mesh, dt, n, xs, neumann_bs=True, c=343):
    V = functionspace(mesh, "Lagrange", 1)


if __name__ == "__main__":

    # Parameters
    mesh_path = Path("electromagnetic/acoustic/tutorial/rectangle.msh")
    dt = 1 / 48000

    mesh, coll, facet_tags = load_mesh(mesh_path, 2)
    u = fem_wave_equation(mesh, dt, 150, dolfin.Point(2, 2), neumann_bc=True)
