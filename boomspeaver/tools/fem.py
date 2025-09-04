from pathlib import Path
from typing import Any, Callable

import gmsh
import matplotlib.pyplot as plt
import numpy as np
import ufl
from dolfinx import fem, io, mesh
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import griddata

from boomspeaver.tools.sampler import spline_interpolate


# Mesh
def load_mesh(
    mesh_input_path: Path | None, dim: int = 2, return_all: bool = False
) -> mesh.Mesh:
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
    if return_all:
        return domain, coll, facet_tags
    else:
        return domain


def get_default_rectangle(x: int = 10, y: int = 10, mesh_size: int = 500) -> mesh.Mesh:
    """Create default membrane for testing."""
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-x / 2, -y / 2]), np.array([x / 2, y / 2])],
        [mesh_size, mesh_size],
        mesh.CellType.triangle,
    )
    return domain


def create_circular_mesh(
    radius: float, mesh_size: float, output_dir: Path
) -> mesh.Mesh:
    """Create circle geometry, mesh, save in output directory and return mesh."""
    assert isinstance(output_dir, Path)
    output_path = str(output_dir / "circle.msh")
    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("circle")

    center = (0, 0, 0)
    disk = gmsh.model.occ.addDisk(*center, radius, radius)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [disk], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Membrane")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.mesh.generate(2)

    gmsh.write(str(output_path))
    if gmsh.isInitialized():
        gmsh.finalize()

    domain = load_mesh(Path(output_path))
    return domain


def create_annulus_mesh(
    outer_radius: float, inner_radius: float, mesh_size: float, output_path: Path
):
    """Create a 2D ring mesh (outer circle with cut-out inner circle)."""
    assert isinstance(output_path, Path)
    output_path = str(output_path.with_suffix(".msh"))

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.model.add("annulus")

    center = (0, 0, 0)
    outer = gmsh.model.occ.addDisk(*center, outer_radius, outer_radius)
    inner = gmsh.model.occ.addDisk(*center, inner_radius, inner_radius)

    ring, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()

    ring_tag = ring[0][1]
    gmsh.model.addPhysicalGroup(2, [ring_tag], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Membrane")

    boundary_curves = gmsh.model.getBoundary(ring, oriented=False, recursive=True)

    curve_tags = [c[1] for c in boundary_curves]
    gmsh.model.addPhysicalGroup(1, curve_tags, tag=2)
    gmsh.model.setPhysicalName(1, 2, "Boundary")

    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.mesh.generate(2)

    gmsh.write(output_path)
    if gmsh.isInitialized():
        gmsh.finalize()

    domain = load_mesh(Path(output_path))
    return domain


def create_circular_3d_membrane_mesh(
    radius: float, mesh_size: float, thickness: float, output_dir: Path
) -> mesh.Mesh:
    """Create 3D circular membrane mesh by extruding 2D disk, save and load as dolfinx mesh."""

    assert isinstance(output_dir, Path)
    output_path = str(output_dir / "circular_membrane.msh")

    if not gmsh.isInitialized():
        gmsh.initialize()

    gmsh.model.add("circular_membrane")

    center = (0, 0, 0)
    disk = gmsh.model.occ.addDisk(*center, radius, radius)
    gmsh.model.occ.synchronize()

    extruded_entities = gmsh.model.occ.extrude([(2, disk)], 0, 0, thickness)
    gmsh.model.occ.synchronize()

    volumes = [ent for ent in extruded_entities if ent[0] == 3]
    gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1)
    gmsh.model.setPhysicalName(3, 1, "MembraneVolume")

    surfaces = [ent for ent in extruded_entities if ent[0] == 2]
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=2)
    gmsh.model.setPhysicalName(2, 2, "MembraneSurface")

    points = gmsh.model.getEntities(0)
    gmsh.model.mesh.setSize(points, mesh_size)

    gmsh.model.mesh.generate(3)

    gmsh.write(output_path)

    if gmsh.isInitialized():
        gmsh.finalize()

    domain = load_mesh(Path(output_path), dim=3)
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


def load_xdmf(xdmf_path: Path) -> io.utils.XDMFFile:
    """Load xdmf state from file."""
    assert isinstance(xdmf_path, Path)
    assert xdmf.exists()
    raise NotImplementedError


# FEM
def initialize_fem_function(V: fem.FunctionSpace, name: str) -> fem.function.Function:
    """Initialize FEM Function."""
    func = fem.Function(V)
    func.name = name
    return func


def get_surface_cords(
    domain: mesh.Mesh, V: fem.FunctionSpace, facet_tags: mesh.meshtags, tag: int
) -> np.ndarray:
    """
    Calculate the geometric coordinations of a tagged surface in a mesh (from domain, facet_tags and V).
    """
    tdim = domain.topology.dim
    assert tdim == 3
    facet_dim = tdim - 1

    # Find all facets with this tag
    surface_facets = np.where(facet_tags.values == tag)[0]
    if len(surface_facets) == 0:
        raise ValueError(f"No facets found with tag {tag}")

    surface_dofs = fem.locate_dofs_topological(
        V, facet_dim, facet_tags.indices[surface_facets]
    )

    coords = V.tabulate_dof_coordinates()[surface_dofs]
    return coords


def get_surface_edge_coords(
    domain: mesh.Mesh, facet_tags: mesh.meshtags, tag: int
) -> np.ndarray:
    """
    Extract coordinates of vertices along the edges of a tagged surface.
    """
    tdim = domain.topology.dim

    assert tdim == 3
    fdim = tdim - 1

    surface_facets = np.where(facet_tags.values == tag)[0]
    if len(surface_facets) == 0:
        raise ValueError(f"No facets found with tag {tag}")

    facet_indices = facet_tags.indices[surface_facets]

    domain.topology.create_connectivity(fdim, 0)
    facet_vertices = []
    for f in facet_indices:
        facet_vertices.extend(domain.topology.connectivity(fdim, 0).links(f))

    unique_vertices = np.unique(facet_vertices)

    return domain.geometry.x[unique_vertices]


def compute_polar_coordinates(V: fem.FunctionSpace, return_degrees: bool = True):
    """
    Compute polar coordinates (r, phi) of the DOFs in a function space,
    centered at the mean of all DOFs.
    """
    coords = V.tabulate_dof_coordinates()
    center = np.mean(coords, axis=0)

    x = coords[:, 0] - center[0]
    y = coords[:, 1] - center[1]

    r = np.sqrt(x**2 + y**2)
    r_max = np.max(r)
    print(r_max)
    r = 100 * r / r_max
    phi = np.arctan2(y, x)

    if return_degrees:
        phi = np.degrees(phi)

    return r, phi, center, np.max(r)


def precalculate_surface_dofs(
    V: fem.FunctionSpace, domain: mesh.Mesh, facet_tags: mesh.MeshTags, surface_tag: int
):
    assert isinstance(surface_tag, int)
    surface_coords = get_surface_cords(
        domain=domain, facet_tags=facet_tags, V=V, tag=surface_tag
    )
    center = np.mean(surface_coords, axis=0)

    facets = facet_tags.find(surface_tag)
    dofs_on_surface = fem.locate_dofs_topological(V, domain.topology.dim - 1, facets)

    x = surface_coords[:, 0] - center[0]
    y = surface_coords[:, 2] - center[2]
    r_target = np.sqrt(x**2 + y**2)
    r_target = 100 * r_target / np.max(r_target)
    phi_target = np.arctan2(y, x)
    x_target = r_target * np.cos(phi_target)
    y_target = r_target * np.sin(phi_target)
    return dofs_on_surface, x_target, y_target


def interpolate_surface_values(
    membrane_displacement: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
    plot_flag: bool = False,
    output_path: Path | None = None,
):
    """
    Interpolate values from a source surface (in polar coords) to a target surface (Cartesian coords).
    """
    values_source = membrane_displacement[:, 0]
    r_source = membrane_displacement[:, 1]
    phi_source = membrane_displacement[:, 2]

    x_source = r_source * np.cos(phi_source)
    y_source = r_source * np.sin(phi_source)

    values_target = griddata(
        points=np.stack([x_source, y_source], axis=1),
        values=values_source,
        xi=np.stack([x_target, y_target], axis=1),
        method="nearest",
    )
    if plot_flag and output_path:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        sc1 = axs[0].scatter(x_source, y_source, s=0.1, c=values_source, cmap="viridis")
        axs[0].set_title("Source values")
        plt.colorbar(sc1, ax=axs[0])

        sc2 = axs[1].scatter(x_target, y_target, s=0.1, c=values_target, cmap="viridis")
        axs[1].set_title("Interpolated values on target")
        plt.colorbar(sc2, ax=axs[1])

        plt.tight_layout()
        plt.axis("equal")

        plt.savefig(str(Path(output_path.parent, "interpolation_debug.png")), dpi=150)
        plt.close()

    return values_target


def wave_equation(
    V: fem.FunctionSpace,
    domain: mesh.Mesh,
    u0: fem.function.Function,
    u1: fem.function.Function,
    dt: float,
    c: int = 343,
    force: fem.function.Function | None = None,
    damping_coefficient: float | None = None,
    surface_force_id: int | None = None,
    facet_tags: mesh.MeshTags | None = None,
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
        if surface_force_id and isinstance(surface_force_id, int):
            ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
            l += force * v * ds(surface_force_id)
        else:
            l += force * v * ufl.dx
    if damping_coefficient is not None:
        gamma = fem.Constant(domain, PETSc.ScalarType(damping_coefficient))
        a += gamma * ufl.inner(u, v) * ufl.ds
    return a, l


def convert_to_form(input_forms: list[ufl.form.Form]) -> list[fem.forms.Form]:
    """Convert ufl form to dolfinx form for time simulation."""
    assert isinstance(input_forms, list)
    assert isinstance(input_forms[0], ufl.form.Form)
    return [fem.form(i) for i in input_forms]


# Boundary Conditions
def set_bcs(
    name: str,
    domain: mesh.Mesh,
    V: fem.FunctionSpace,
    facet_tags: mesh.MeshTags | None = None,
    surface_tags: list[int] | None = None,
):
    """Set boundary conditions on all surfaces."""
    if name is None:
        return None
    assert name in {"neumann", "dirichlet", "robin"}

    fdim = domain.topology.dim - 1
    if not surface_tags or not facet_tags:
        boundary_facets = mesh.locate_entities_boundary(
            domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
    else:
        boundary_facets = []
        for tag in surface_tags:
            facets = facet_tags.find(tag)
            boundary_facets.append(facets)
        boundary_facets = np.hstack(boundary_facets)

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

    def ring_region(
        self,
        x,
        r: float = 0.001,
        width: float = 0.001,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        x0, y0 = center
        radius = np.sqrt((x[0] - x0) ** 2 + (x[1] - y0) ** 2)
        return np.logical_and(radius > r - width / 2, radius < r + width / 2)

    def circle_region(
        self,
        x,
        r: float = 0.001,
        center: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """
        Returns 1.0 for points inside a filled circle of radius r, 0.0 outside.
        """
        x0, y0 = center
        radius = np.sqrt((x[0] - x0) ** 2 + (x[1] - y0) ** 2)
        inside = radius <= r
        return np.where(inside, 1.0, 0.0)

    def radial_spline_profile(
        self,
        x,
        values: np.ndarray,
        center: tuple[float, float] = (0.0, 0.0),
        r_min: float = 0.0,
        r_max: float = 1.0,
    ) -> np.ndarray:
        """
        Radial profile evaluated via spline interpolation at runtime.
        `values` must be passed at each call to allow dynamic updates.
        """
        if values is None:
            raise ValueError("`values` must be provided at each call.")

        x0, y0 = center
        radius = np.sqrt((x[0] - x0) ** 2 + (x[1] - y0) ** 2)

        r_norm = (radius - r_min) / (r_max - r_min)
        r_norm = np.clip(r_norm, 0, 1)

        spline = spline_interpolate(values=values)
        return spline(r_norm)

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
