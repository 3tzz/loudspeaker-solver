from pathlib import Path

import numpy as np
from dolfinx import io, mesh


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


def get_default_rectangle():
    nx, ny = 200, 200
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-2, -2]), np.array([2, 2])],
        [nx, ny],
        mesh.CellType.triangle,
    )
    return domain
