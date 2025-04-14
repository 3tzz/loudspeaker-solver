import matplotlib as mpl
import numpy as np
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

# Define temporal parameters
t = 0  # Start time
# T = 100.0  # Final time
T = 1.0  # Final time
num_steps = 100
dt = T / num_steps  # time step size

# Define mesh
nx, ny = 200, 200
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([-2, -2]), np.array([2, 2])],
    [nx, ny],
    mesh.CellType.triangle,
)
V = fem.functionspace(domain, ("Lagrange", 1))


# Create initial condition
def initial_condition(x, a=5):
    return np.exp(-a * (x[0] ** 2 + x[1] ** 2))


def spatial_profile(x, R=0.5, width=0.1):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    ring = np.logical_and(r > (R - width), r < (R + width))
    return np.where(ring, 1.0, 0.0)


def force_sine(t, amplitude=-1.0, frequency=5.0):
    return amplitude * np.sin(2 * np.pi * frequency * t)


u_n = fem.Function(V)
u_n.name = "u_n"
# u_n.interpolate(initial_condition_circle)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
)
bc = fem.dirichletbc(
    PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V
)

xdmf = io.XDMFFile(
    domain.comm,
    "/home/shared/electromagnetic/acoustic/tutorial/output/diffusion.xdmf",
    "w",
)
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
# uh.interpolate(initial_condition_circle)
xdmf.write_function(uh, t)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

f = fem.Function(V)
f.name = "force"
# f = fem.Constant(domain, PETSc.ScalarType(1))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
# a = u * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
# L = (u_n + dt * (100 / 79) * f) * v * ufl.dx
L = (u_n + dt * f) * v * ufl.dx
# L = (u_n + f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form, bcs=[bc])
A.assemble()
b = create_vector(linear_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

for i in range(num_steps):
    t += dt

    f.interpolate(lambda x: force_sine(t) * spatial_profile(x, width=0.2))
    # f.interpolate(
    #     lambda x: spatial_sinusoidal_force(x, t, amplitude=1.0, frequency=10.0)
    # )
    # if i == 50:
    # f.value = PETSc.ScalarType(-1)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)

    # Apply Dirichlet boundary condition to the vector
    apply_lifting(b, [bilinear_form], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bc])

    # Solve linear problem
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)
xdmf.close()
