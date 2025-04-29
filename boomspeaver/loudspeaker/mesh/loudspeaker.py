from pathlib import Path

import gmsh

from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.data import get_repo_dir, get_value_from_dict, load_json_file


def calculate_mesh_size(frequency_hz, speed_of_sound=343, elements_per_wavelength=10):
    wavelength = speed_of_sound / frequency_hz
    mesh_size = wavelength * 1000 / elements_per_wavelength  # to [mm]
    return mesh_size


def create_lsroom_2d(
    room_length: int,
    room_width: int,
    box_length: int,
    box_width: int,
    membrane: int,
    output_path: Path,
    max_frequency: float,
    focus_source: int = 1000,
):
    print(room_length)
    print(room_width)
    print(box_length)
    print(box_width)
    print(membrane)

    gmsh.initialize()
    mesh_size_focused = calculate_mesh_size(max_frequency)
    mesh_size = 1000
    length_sym = room_length / 2
    width_sym = room_width / 2
    membrane_thickness = 1

    # Define the points for the rectangle (room dimensions)
    gmsh.model.geo.addPoint(0, 0, 0, mesh_size)  # Point 1
    gmsh.model.geo.addPoint(length_sym - (box_width / 2), 0, 0, mesh_size)  # Point 2
    gmsh.model.geo.addPoint(
        length_sym - (box_width / 2), box_length, 0, mesh_size
    )  # Point 3
    gmsh.model.geo.addPoint(
        length_sym - (membrane / 2), box_length, 0, mesh_size
    )  # Point 4
    gmsh.model.geo.addPoint(
        length_sym - (membrane / 2), box_length + membrane_thickness, 0, mesh_size
    )  # Point 5
    gmsh.model.geo.addPoint(
        length_sym + (membrane / 2), box_length + membrane_thickness, 0, mesh_size
    )  # Point 6
    gmsh.model.geo.addPoint(
        length_sym + (membrane / 2), box_length, 0, mesh_size
    )  # Point 7
    gmsh.model.geo.addPoint(
        length_sym + (box_width / 2), box_length, 0, mesh_size
    )  # Point 8
    gmsh.model.geo.addPoint(length_sym + (box_width / 2), 0, 0, mesh_size)  # Point 9
    gmsh.model.geo.addPoint(room_length, 0, 0, mesh_size)  # Point 10
    gmsh.model.geo.addPoint(room_length, room_width, 0, mesh_size)  # Point 11
    gmsh.model.geo.addPoint(0, room_width, 0, mesh_size)  # Point 12

    gmsh.model.geo.addPoint(length_sym - (focus_source / 2), 0, 0)  # Point 13
    gmsh.model.geo.addPoint(
        length_sym - (focus_source / 2), box_length + focus_source, 0
    )  # Point 14
    gmsh.model.geo.addPoint(
        length_sym + (focus_source / 2), box_length + focus_source, 0
    )  # Point 15
    gmsh.model.geo.addPoint(length_sym + (focus_source / 2), 0, 0)  # Point 16

    # Define the lines (walls of the room)
    gmsh.model.geo.addLine(1, 13)  # Line 1

    gmsh.model.geo.addLine(16, 9)  # Line 2
    gmsh.model.geo.addLine(9, 8)  # Line 3
    gmsh.model.geo.addLine(8, 7)  # Line 4
    gmsh.model.geo.addLine(7, 6)  # Line 5
    gmsh.model.geo.addLine(6, 5)  # Line 6
    gmsh.model.geo.addLine(5, 4)  # Line 7
    gmsh.model.geo.addLine(4, 3)  # Line 8
    gmsh.model.geo.addLine(3, 2)  # Line 9
    gmsh.model.geo.addLine(2, 13)  # Line 10
    gmsh.model.geo.addLine(16, 10)  # Line 11
    gmsh.model.geo.addLine(10, 11)  # Line 12
    gmsh.model.geo.addLine(11, 12)  # Line 13
    gmsh.model.geo.addLine(12, 1)  # Line 14
    gmsh.model.geo.addLine(13, 14)  # Line 15
    gmsh.model.geo.addLine(14, 15)  # Line 16
    gmsh.model.geo.addLine(15, 16)  # Line 17
    gmsh.model.geo.addLine(16, 13)  # Line 18

    # Define the loop to create the surface (the room)

    # Whole surface
    # gmsh.model.geo.addCurveLoop([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    # gmsh.model.geo.addPlaneSurface([1])

    # Default part
    gmsh.model.geo.addCurveLoop([1, 15, 16, 17, 11, 12, 13, 14])
    gmsh.model.geo.addPlaneSurface([1])

    # Focused part
    # gmsh.model.geo.addCurveLoop([18, 17, 16, 15])  # Loop through the lines
    gmsh.model.geo.addCurveLoop(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17]
    )  # Loop through the lines
    gmsh.model.geo.addPlaneSurface([2])  # Create the surface inside the loop

    # Focused size mesh using fields
    field = gmsh.model.mesh.field
    field.add("Distance", 1)
    field.setNumbers(1, "EdgesList", [5])  # for diaphragm

    field.add("Threshold", 2)
    field.setNumber(2, "InField", 1)
    field.setNumber(2, "SizeMin", mesh_size_focused)
    field.setNumber(2, "SizeMax", mesh_size_focused)
    field.setNumber(2, "DistMin", mesh_size_focused)
    field.setNumber(2, "DistMax", mesh_size_focused)

    field.add("Restrict", 3)
    field.setNumber(3, "InField", 2)
    field.setNumbers(3, "SurfacesList", [2])  # <-- for surface

    field.setAsBackgroundMesh(3)

    gmsh.model.addPhysicalGroup(2, [1], 1, name="air")
    gmsh.model.addPhysicalGroup(2, [2], 2, name="very_important_air")
    gmsh.model.addPhysicalGroup(1, [5], 3, name="membrane")

    # Synchronize the model and generate the mesh
    gmsh.model.geo.synchronize()

    # Generate the mesh
    gmsh.model.mesh.generate(2)
    # Write the mesh to a file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(output_path))

    # Finalize the Gmsh session
    gmsh.finalize()


def main():

    # print(get_repo_dir(run_type="docker"))
    room_path = get_repo_dir(run_type="python") / "examples/room_geometry.json"
    loudspeaker_path = (
        get_repo_dir(run_type="python") / "examples/prv_audio_6MB400_8ohm.json"
    )

    output_path = Path(__file__).resolve().parent / "output/room_ls_2d.msh"
    room = load_json_file(room_path)
    loudspeaker = Loudspeaker.from_json(loudspeaker_path)

    # Get room dimensions
    room_length = room["room"]["length"]
    room_width = room["room"]["width"]
    box_length = room["box"]["length"]
    box_width = room["box"]["width"]
    membrane = loudspeaker.diaphragm.diameter
    max_frequency = loudspeaker.frequency_response["max"]

    create_lsroom_2d(
        room_length=int(room_length),
        room_width=int(room_width),
        box_length=int(box_length),
        box_width=int(box_width),
        membrane=int(membrane),
        max_frequency=max_frequency,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
