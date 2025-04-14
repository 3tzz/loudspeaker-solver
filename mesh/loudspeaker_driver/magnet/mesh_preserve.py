import os
import sys
from pathlib import Path

import gmsh

sys.path.insert(0, "/home/freetzz/repo/fenics/mesh/tools/")
from mesh_tools import (
    apply_mesh_sizing,
    configure_mesh,
    define_new_volume,
    define_physical_groups,
    generate_mesh,
    load_gmsh,
)


def mesh(input_path: Path, output_directory: Path) -> None:
    """Main function to process meshing."""
    if not input_path.exists():
        raise ValueError(f"Provided path doesn't exist: {input_path}")
    if not output_directory.exists():
        raise ValueError(f"Provided output directory doesn't exist: {output_directory}")

    output_path = str(output_directory / input_path.with_suffix(".msh").name)
    name = input_path.stem

    # Mesh settings
    # metrics = "M"
    metrics = None
    mesh_size_max = None
    mesh_size_min = None

    # [m]
    mesh_size_max = 0.5
    mesh_size_min = 0.01
    # [mm]
    mesh_size_max = mesh_size_max * 1000  # 500
    mesh_size_min = mesh_size_min * 1000  # 10

    # Surface definitions
    volumes = {
        "air": [1],
        "magnet": [2],
    }

    walls = {
        "wall_left": [1],
        "wall_front": [2],
        "wall_bottom": [3],
        "wall_rear": [4],
        "wall_top": [5],
        "wall_right": [6],
    }
    # Room
    surfaces_groups = {
        "magnet_top": [8],
        "magnet_side_external": [7],
        "magnet_bottom": [9],
        "magnet_side_internal": [10],
    }
    # MagnetInRoom
    # surfaces_groups = {
    #     "magnet_top": [1],
    #     "magnet_side_external": [2],
    #     "magnet_bottom": [3],
    #     "magnet_side_internal": [4],
    #     "magnet_top": [11],
    #     "magnet_side_external": [12],
    #     "magnet_bottom": [13],
    #     "magnet_side_internal": [14],
    # }

    # Initialize and configure GMSH
    load_gmsh(str(input_path), name, metrics, mesh_size_max, mesh_size_min)

    # configure_mesh(mesh_size_max, mesh_size_min)
    surfaces = [i[0] for i in surfaces_groups.values()]
    define_new_volume(surfaces)
    # apply_mesh_sizing(mesh_size_min, mesh_size_max, surfaces_groups)
    define_physical_groups(volumes, surfaces_groups | walls, name)

    # Generate and save mesh
    generate_mesh(output_path)

    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    repo_path = "/home/freetzz/repo/fenics/geometric/tools/export/"
    input_path = Path(repo_path, "Room.brep")
    # input_path = Path(repo_path, "MagnetInRoom.brep")
    output_directory = Path(os.path.dirname(os.path.abspath(__file__)), "export")
    output_directory.mkdir(exist_ok=True, parents=True)
    mesh(input_path, output_directory)
