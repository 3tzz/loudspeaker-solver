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
    mesh_size_max = 0.3
    mesh_size_max = 0.5

    # Initialize gmsh
    gmsh.initialize(sys.argv)
    gmsh.model.add(name)
    gmsh.option.setString(
        "Geometry.OCCTargetUnit", metrics
    )  # NOTE:define units [meters]
    gmsh.merge(input_path)

    # print(gmsh.model.getEntities(2))  # List all surface IDs
    walls = [1, 6, 8, 9]
    floor = [2]
    celling = [7]
    membrane = [12]
    loudspeaker_box = [3, 4, 5, 10]
    ls_mem_edge = [11]

    gmsh.option.setNumber(
        "Mesh.MeshSizeMax", mesh_size_max
    )  # NOTE: basic mesh size value

    # Get all surfaces (2D)
    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.occ.synchronize()

    # Define Surface IDs
    membrane_surface = 12  # Membrane surface ID
    # other_surface_ids = [
    #     s[1] for s in gmsh.model.getEntities(dim=2) if s[1] != membrane_surface
    # ]

    # Create a field for mesh size control
    # 1. Define a constant field for the membrane surface
    # mesh_size_membrane = 0.002
    mesh_size_membrane = 0.009
    mesh_size_membrane = 0.015
    id = 1
    set_surface_mesh_size(mesh_size_membrane, id, mesh_size_max, surfaces=membrane)

    # 2. Define a constant field for the loudspeaker_box surface
    mesh_size_lsbox = 0.08
    id = 2
    set_surface_mesh_size(mesh_size_lsbox, id, mesh_size_max, surfaces=loudspeaker_box)

    # Combine fields visibility
    gmsh.model.mesh.field.add("Min", 3)
    gmsh.model.mesh.field.setNumbers(3, "FieldsList", [1, 2])

    # gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 5)
    gmsh.model.mesh.field.setAsBackgroundMesh(3)
    # raise
    # # Physical setup
    # gmsh.model.addPhysicalGroup(
    #     3, [1], 1, name
    # )  # NOTE: arguments ("dimension_of_geometrical_entity", "geometry_id", "mesh_id","output_name")
    #
    # gmsh.model.addPhysicalGroup(
    #     2, [3], 2, "velocity_bc"
    # )  # NOTE: assign boundary conditions for surfaces ("dimension","surface_id","mesh_id", "output_name")
    # gmsh.model.addPhysicalGroup(2, [4], 3, "impedance_bc")

    # Generate mesh
    gmsh.model.mesh.generate(3)  # NOTE: 3 dimensions

    # Save file
    gmsh.write(output_path)

    # Visualize launch the GUI to see the results:
    if "-nopopup" not in sys.argv:
        gmsh.fltk.run()

    gmsh.finalize()


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


if __name__ == "__main__":

    repo_path = "/home/freetzz/repo/fenics/examples/room/"
    input_path = Path(repo_path, "first_ls_in_room_diff_final.step")
    output_directory = Path(root_path)
    mesh(input_path, output_directory)
