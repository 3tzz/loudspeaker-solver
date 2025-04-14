import os
import sys
from pathlib import Path

import gmsh

sys.path.insert(0, "/home/freetzz/repo/fenics/mesh/tools/")
from mesh_tools import apply_mesh_sizing  # set_all_mesh_sizes,
from mesh_tools import (
    apply_mshs_sizing,
    assign_mesh_size,
    configure_mesh,
    define_physical_groups,
    generate_mesh,
    get_from_dict,
    load_gmsh,
    update_mesh_size_for_keys,
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
    mesh_default_size_max = None
    mesh_default_size_min = None

    # [m]
    mesh_default_size_max = 0.007
    mesh_default_size_min = 0.006
    # [mm]
    mesh_default_size_max = mesh_default_size_max * 1000
    mesh_default_size_min = mesh_default_size_min * 1000

    # Mid focus parts
    mesh_mid_size = 0.0007 * 1000
    mesh_mid_size = 0.0001 * 1000

    # Air gap resolution
    crutial_gap_size = 0.000000001 * 1000

    # Surface definitions
    volumes = {
        "magnet": [1],
        "ferro_inside": [2],
        "ferro": [3],
        "air_gap": [4],
        "air": [5],
    }

    walls = {
        "wall_external": [20],
        "wall_top": [21],
        "wall_bottom": [22],
        "air_ferro_top_outer": [23],
        "air_ferro_side_outer": [24],
        "air_xmax_ring": [25],
        "air_ferro_bottom_outer": [26],
        "air_air_gap_top": [27],
    }
    surfaces_groups = {
        "magnet_top": [2],
        "magnet_side_external": [1],
        "magnet_bottom": [3],
        "magnet_side_internal": [4],
        "ferro_inside_external": [5],
        "ferro_inside_top": [6],
        "ferro_inside_bottom": [7],
        "ferro_top_outer": [8],
        "ferro_top_inner": [10],
        "ferro_air_gap": [9],
        "ferro_side_inner": [11],
        "ferro_side_outer": [13],
        "ferro_bottom_inner": [12],
        "ferro_bottom_outer": [14],
        "air_gap_side_external": [15],
        "air_gap_top": [16],
        "air_gap_bottom_outer": [17],
        "air_gap_side_internal": [18],
        "air_gap_bottom_inner": [19],
    }
    all_surfaces = walls | surfaces_groups
    duplicate_surfaces_map = {
        23: 8,
        24: 13,
        26: 14,
        27: 16,
    }
    default_size = {"wall_external", "wall_top", "wall_bottom"}
    crutial_size = {
        "air_gap_side_external",
        "air_gap_top",
        "air_gap_bottom_outer",
        "air_gap_bottom_outer",
        "air_gap_side_internal",
        "air_gap_bottom_inner",
        "ferro_inside_external",
        "ferro_inside_top",
        "ferro_inside_bottom",
        "air_xmax_ring",
        "air_air_gap_top",
        "magnet_side_internal",
        "ferro_top_inner",
    }  # focus on air gap and ferro inside

    mid_focus_size = {
        i for i in all_surfaces.keys() if i not in (default_size | crutial_size)
    }  # main parts

    # Initialize and configure GMSH
    load_gmsh(str(input_path), name, metrics, None, None)
    configure_mesh(mesh_default_size_max, mesh_default_size_min)

    # Set mesh sizes
    msh_sizes = assign_mesh_size(all_surfaces, mesh_default_size_max)
    msh_sizes = update_mesh_size_for_keys(msh_sizes, mid_focus_size, mesh_mid_size)
    msh_sizes = update_mesh_size_for_keys(msh_sizes, crutial_size, crutial_gap_size)
    apply_mshs_sizing(
        msh_sizes,
        mesh_default_size_max,
        get_from_dict(all_surfaces, crutial_size),
    )

    define_physical_groups(volumes, all_surfaces, name)

    # Generate and save mesh
    generate_mesh(output_path)

    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    repo_path = "/home/freetzz/repo/fenics/geometric/loudspeaker_driver/export/"
    input_path = Path(repo_path, "MagnetFerroAirGapInCylinder.brep")
    output_directory = Path(os.path.dirname(os.path.abspath(__file__)), "export")
    output_directory.mkdir(exist_ok=True, parents=True)
    mesh(input_path, output_directory)
