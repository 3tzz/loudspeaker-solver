# pylint: disable=invalid-name

import os
import sys
from pathlib import Path

import gmsh

root_path = os.path.dirname(os.path.abspath(__file__))


def mesh(input_path: Path, output_directory: Path) -> None:

    # Validate
    assert isinstance(output_directory, Path) and isinstance(input_path, Path)
    if not input_path.exists():
        raise ValueError(f"Provided path doesn't exist: {input_path}")
    if not output_directory.exists():
        raise ValueError(f"Provided output directory doesn't exist: {output_directory}")

    # Initialize variables
    output_path = str(input_path.with_suffix(".msh"))
    name = input_path.stem
    input_path = str(input_path)
    metrics = "M"
    mesh_size_max = 0.03

    # Initialize gmsh
    gmsh.initialize(sys.argv)
    gmsh.model.add(name)
    gmsh.option.setString(
        "Geometry.OCCTargetUnit", metrics
    )  # NOTE:define units [meters]
    gmsh.merge(input_path)

    # print(gmsh.model.getEntities(2))  # List all surface IDs
    gmsh.option.setNumber(
        "Mesh.MeshSizeMax", mesh_size_max
    )  # NOTE: basic mesh size value

    # Physical setup
    gmsh.model.addPhysicalGroup(
        3, [1], 1, name
    )  # NOTE: arguments ("dimension_of_geometrical_entity", "geometry_id", "mesh_id","output_name")

    gmsh.model.addPhysicalGroup(
        2, [3], 2, "velocity_bc"
    )  # NOTE: assign boundary conditions for surfaces ("dimension","surface_id","mesh_id", "output_name")
    gmsh.model.addPhysicalGroup(2, [4], 3, "impedance_bc")

    # Generate mesh
    gmsh.model.mesh.generate(3)  # NOTE: 3 dimensions

    # Save file
    gmsh.write(output_path)

    # Visualize launch the GUI to see the results:
    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":

    input_path = Path(root_path, "air_volume.step")
    output_directory = Path(root_path)
    mesh(input_path, output_directory)
