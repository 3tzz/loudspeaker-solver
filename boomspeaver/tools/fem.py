from pathlib import Path
from typing import Any, Callable

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc


# Mesh
def load_mesh(mesh_input_path: Path | None, dim: int = 2) -> mesh.Mesh:
    """Load mesh file from file."""
    assert isinstance(mesh_input_path, Path) or mesh_input_path is None
    assert isinstance(dim, int)

    if not mesh_input_path:
        print("Warning: Default rectangle created as mesh domain.")
        domain = get_default_rectangle()
    else:
        assert mesh_input_path.exists()
        assert mesh_input_path.suffix == ".msh"
        domain, coll, facet_tags, *_ = io.gmshio.read_from_msh(
            str(mesh_input_path), MPI.COMM_WORLD, 0, gdim=dim
        )
    return domain


def get_default_rectangle() -> mesh.Mesh:
    nx, ny = 500, 500
    # nx, ny = 1000, 1000
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-10, -10]), np.array([10, 10])],
        [nx, ny],
        mesh.CellType.triangle,
    )
    return domain


# XDMF
def initialize_xdmf(
    domain: mesh.Mesh,
    output_path: Path,
) -> io.utils.XDMFFile:
    """Initialize XDMF."""
    assert isinstance(domain, mesh.Mesh)
    assert isinstance(output_path, Path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xdmf = io.XDMFFile(
        domain.comm,
        str(output_path.with_suffix(".xdmf")),
        "w",
    )
    xdmf.write_mesh(domain)
    return xdmf


def close_xdmf(xdmf: io.utils.XDMFFile) -> None:
    """Close opened xdmf file."""
    xdmf.close()


# FEM
def initialize_fem_function(V: fem.FunctionSpace, name: str) -> fem.function.Function:
    """Initialize FEM Function."""
    func = fem.Function(V)
    func.name = name
    return func


def wave_equation(
    V: fem.FunctionSpace,
    domain: mesh.Mesh,
    u0: fem.function.Function,
    u1: fem.function.Function,
    dt: float,
    c: int = 343,
    force: fem.function.Function | None = None,
) -> tuple[ufl.form.Form, ufl.form.Form]:
    """Define time dependent wave equation."""
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (
        ufl.inner(u, v) * ufl.dx
        + fem.Constant(domain, PETSc.ScalarType(dt**2 * c**2))
        * ufl.inner(ufl.nabla_grad(u), ufl.nabla_grad(v))
        * ufl.dx
    )
    l = 2 * u1 * v * ufl.dx - u0 * v * ufl.dx
    if force is not None:
        l += force * v * ufl.dx
    return a, l


def convert_to_form(input_forms: list[ufl.form.Form]) -> list[fem.forms.Form]:
    """Convert ufl form to dolfinx form for time simulation."""
    assert isinstance(input_forms, list)
    assert isinstance(input_forms[0], ufl.form.Form)
    return [fem.form(i) for i in input_forms]


# Boundary Conditions
def set_bcs(name: str, domain: mesh.Mesh, V: fem.FunctionSpace):
    """Set boundary conditions on all surfaces."""
    assert name in {"neumann", "dirichlet", "robin"}
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )

    if name == "neumann":
        return None
    elif name == "dirichlet":
        return [
            fem.dirichletbc(
                PETSc.ScalarType(0),
                fem.locate_dofs_topological(V, fdim, boundary_facets),
                V,
            )
        ]
    else:
        raise NotImplementedError(f"Unknown boundaries type {name}.")


# Shape
class Shape:
    def __init__(self, name: str, kwargs: dict = None, verbose: bool = False) -> None:
        self.verbose = verbose
        self.kwargs = kwargs or {}
        self.shape = self._get_method_via_name(name)

    def _get_method_via_name(self, name: str) -> Callable:
        """Return a bound shape method by name with kwargs ready to apply."""
        assert isinstance(name, str)

        method = getattr(self, name, None)
        if method is None:
            raise AttributeError(f"Shape method '{name}' not found.")
        if not callable(method):
            raise TypeError(f"Attribute '{name}' exists but is not callable.")

        if self.verbose:
            print(f"[Shape] Using method: {name} with kwargs: {self.kwargs}")
        return lambda x: method(x, **self.kwargs)

    def ring_spatial_profile(
        self,
        x,
        r: float = 0.001,
        width: float = 0.001,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Spatial ring shape in 2D dimension."""
        x0, y0 = center
        radius = np.sqrt((x[0] - x0) ** 2 + (x[1] - y0) ** 2)
        ring = np.logical_and(radius > (r - width), radius < (r + width))
        if self.verbose:
            print(f"[Shape] Ring inside domain: {np.any(ring)}")
        return np.where(ring, 1.0, 0.0)

    def square_spatial_profile(
        self,
        x,
        center: tuple[float, float] = (0.0, 0.0),
        shape: tuple[float, float] = (0.1, 0.1),
    ) -> np.ndarray:
        """Spatial ring shape in 2D dimension."""
        x0, y0 = center
        width, height = shape
        mask = np.logical_and.reduce(
            [
                x[0] > (x0 - width),
                x[0] < (x0 + width),
                x[1] > (y0 - height),
                x[1] < (y0 + height),
            ]
        )
        if self.verbose:
            print(f"[Shape] Square inside domain: {np.any(mask)}")
        return mask.astype(np.float64)
