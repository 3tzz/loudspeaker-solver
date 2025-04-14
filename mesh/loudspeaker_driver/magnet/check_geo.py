import os
import sys
from pathlib import Path

import gmsh

sys.path.insert(0, "/home/freetzz/repo/fenics/mesh/tools/")
from mesh_tools import check_volumes_and_surfaces, configure_mesh, load_gmsh


def mesh(input_path: Path, output_directory: Path) -> None:
    """Main function to process meshing."""
    if not input_path.exists():
        raise ValueError(f"Provided path doesn't exist: {input_path}")
    name = input_path.stem

    # Mesh settings
    metrics = "M"
    mesh_size_max = 0.5
    mesh_size_min = 0.01

    # Initialize and configure GMSH
    load_gmsh(str(input_path), name, metrics)
    configure_mesh(mesh_size_max, mesh_size_min)
    check_volumes_and_surfaces()
    raise


if __name__ == "__main__":
    repo_path = "/home/freetzz/repo/fenics/geometric/loudspeaker_driver/export/"
    input_path = Path(repo_path, "MagnetInRoom.step")
    output_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    mesh(input_path, output_directory)
