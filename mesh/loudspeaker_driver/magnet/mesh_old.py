# pylint: disable=invalid-name

import os
import sys
from pathlib import Path

import gmsh

root_path = os.path.dirname(os.path.abspath(__file__))


def set_surface_mesh_size(
    mesh_size: float,
    field_id: int,
    default_mesh_size: float,
    surfaces: list[int],
):
    assert isinstance(mesh_size, float)
    assert isinstance(field_id, int)
    assert isinstance(default_mesh_size, float)
    assert isinstance(surfaces, list)
    id = field_id
    gmsh.model.mesh.field.add("Constant", id)
    gmsh.model.mesh.field.setNumbers(id, "SurfacesList", surfaces)
    gmsh.model.mesh.field.setNumber(id, "VIn", mesh_size)
    gmsh.model.mesh.field.setNumber(id, "VOut", default_mesh_size)


def mesh(input_path: Path, output_directory: Path) -> None:

    # Validate
    assert isinstance(output_directory, Path) and isinstance(input_path, Path)
    if not input_path.exists():
        raise ValueError(f"Provided path doesn't exist: {input_path}")
    if not output_directory.exists():
        raise ValueError(f"Provided output directory doesn't exist: {output_directory}")

    # Initialize variables
    output_path = str(Path(output_directory, input_path.with_suffix(".msh").name))
    name = input_path.stem
    input_path = str(input_path)
    metrics = "M"
    mesh_size_max = 0.5
    mesh_size_min = 0.001
    mesh_size_min = 0.01

    # Initialize gmsh
    gmsh.initialize(sys.argv)
    gmsh.model.add(name)
    gmsh.option.setString(
        "Geometry.OCCTargetUnit", metrics
    )  # NOTE:define units [meters]
    gmsh.merge(input_path)

    # print(gmsh.model.getEntities(2))  # List all surface IDs
    volume = [1]
    walls = [1, 2, 3, 4, 5, 6]
    magnet_top = [8]
    magnet_bottom = [9]
    magnet_side_external = [7]
    magnet_side_internal = [10]

    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)

    mesh_size_magnet = mesh_size_min
    id = 1
    set_surface_mesh_size(
        mesh_size_magnet,
        id,
        mesh_size_max,
        surfaces=magnet_top,
    )
    id = 2
    set_surface_mesh_size(
        mesh_size_magnet,
        id,
        mesh_size_max,
        surfaces=magnet_bottom,
    )
    id = 3
    set_surface_mesh_size(
        mesh_size_magnet,
        id,
        mesh_size_max,
        surfaces=magnet_side_internal,
    )
    id = 4
    set_surface_mesh_size(
        mesh_size_magnet,
        id,
        mesh_size_max,
        surfaces=magnet_side_external,
    )

    # Combine fields visibility
    gmsh.model.mesh.field.add("Min", 10)
    gmsh.model.mesh.field.setNumbers(10, "FieldsList", [1, 2, 3, 4])

    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.field.setAsBackgroundMesh(10)

    # Physical setup
    gmsh.model.addPhysicalGroup(
        3, [1], 1, name
    )  # NOTE: arguments ("dimension_of_geometrical_entity", "geometry_id", "mesh_id","output_name")

    gmsh.model.addPhysicalGroup(2, magnet_top, 2, "magnet_top")
    gmsh.model.addPhysicalGroup(2, magnet_bottom, 3, "magnet_bottom")
    gmsh.model.addPhysicalGroup(2, magnet_side_external, 4, "magnet_side_external")
    gmsh.model.addPhysicalGroup(2, magnet_side_internal, 5, "magnet_side_internal")

    # Generate mesh
    gmsh.model.mesh.generate(3)  # NOTE: 3 dimensions

    # Save file
    gmsh.write(output_path)

    # Visualize launch the GUI to see the results:
    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":

    repo_path = "/home/freetzz/repo/fenics/geometric/loudspeaker_driver/export/"
    input_path = Path(repo_path, "ls-magnet_in_studio.step")
    output_directory = Path(root_path)
    mesh(input_path, output_directory)
