# pylint: disable=invalid-name


import json
from typing import Any

import FreeCAD as App
import numpy as np
import Part


def cut_parts_from_loudspeaker(loudspeaker_parts, base_part, parts_to_cut):
    doc = App.activeDocument()

    if base_part not in loudspeaker_parts:
        raise ValueError(f"{base_part} not found in loudspeaker_parts.")

    base_shape = loudspeaker_parts[base_part]
    cut = base_shape

    for part in parts_to_cut:
        if part not in loudspeaker_parts:
            raise ValueError(f"{part} not found in loudspeaker_parts.")

        part_shape = loudspeaker_parts[part]
        cut = cut.cut(part_shape)

    cut_object = doc.addObject("Part::Feature", f"{base_part}_cut")
    cut_object.Shape = cut
    doc.recompute()


def remove_parts(parts_to_remove):
    """
    Remove specified parts from.
    """
    doc = App.activeDocument()
    for obj in doc.Objects:
        if obj.Name in parts_to_remove:
            doc.removeObject(obj.Name)
    doc.recompute()


def load_json_file(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    return loaded_data


class LoudspeakerDriver:
    def __init__(
        self,
        membrane_r1,
        membrane_r2,
        membrane_h,
        coil_r_outer,
        coil_r_inner,
        coil_h,
        coil_embed,
        magnet_r_outer,
        magnet_r_inner,
        magnet_h,
        ferro_thickness,
        air_gap,
        split_air_field=False,
        membrane_is_cone=True,  # Add this flag to choose between cone or flat
        merge_ferro=False,  # Add a flag to merge ferro parts
        cylinder_r_outer=None,  # Add this parameter for the cylinder radius
        cylinder_h=None,  # Add this parameter for the cylinder height
    ):
        self.membrane_r1 = membrane_r1
        self.membrane_r2 = membrane_r2
        self.membrane_h = membrane_h
        self.coil_r_outer = coil_r_outer
        self.coil_r_inner = coil_r_inner
        self.coil_h = coil_h
        self.coil_embed = coil_embed
        self.magnet_r_outer = magnet_r_outer
        self.magnet_r_inner = magnet_r_inner
        self.magnet_h = magnet_h
        self.ferro_thickness = ferro_thickness
        self.air_gap = air_gap
        self.split_air_field = split_air_field
        self.membrane_is_cone = membrane_is_cone
        self.merge_ferro = merge_ferro
        self.cylinder_r_outer = cylinder_r_outer
        self.cylinder_h = cylinder_h


def create_membrane(schema):
    if schema.membrane_is_cone:
        # Create membrane as a cone
        membrane_outer = Part.makeCone(
            schema.membrane_r2, schema.membrane_r1, schema.membrane_h
        )

        # Inner cone (smaller by 1mm thickness)
        thickness = 1.0  # [mm]
        membrane_inner = Part.makeCone(
            schema.membrane_r2 - thickness,
            schema.membrane_r1 - thickness,
            schema.membrane_h + thickness,
        )

        # Subtract inner cone from outer cone
        membrane = membrane_outer.cut(membrane_inner)

    else:
        membrane_h = (
            1  # default value not provided maybe should be smaller to check TODO
        )
        # Create membrane as a flat circle
        membrane = Part.makeCylinder(
            schema.membrane_r1, membrane_h
        )  # Use the larger radius for a flat circle
    membrane.translate(
        App.Vector(
            0,
            0,
            2 * schema.ferro_thickness
            + schema.magnet_h
            + ((schema.coil_h - schema.air_gap) / 2),
        )
    )
    return membrane


def create_coil(schema):
    outer_cylinder = Part.makeCylinder(schema.coil_r_outer, schema.coil_h)
    inner_cylinder = Part.makeCylinder(schema.coil_r_inner, schema.coil_h)
    coil = outer_cylinder.cut(inner_cylinder)
    coil.translate(
        App.Vector(
            0,
            0,
            2 * schema.ferro_thickness
            + schema.magnet_h
            - (schema.coil_h - ((schema.coil_h - schema.air_gap) / 2)),
        )
    )
    return coil


def create_magnet(schema):
    outer_magnet = Part.makeCylinder(schema.magnet_r_outer, schema.magnet_h)
    inner_magnet = Part.makeCylinder(schema.magnet_r_inner, schema.magnet_h)
    magnet = outer_magnet.cut(inner_magnet)
    magnet.translate(App.Vector(0, 0, schema.ferro_thickness))
    return magnet


def create_ferro_back(schema):
    back = Part.makeCylinder(
        schema.magnet_r_outer + schema.ferro_thickness, schema.ferro_thickness
    )
    return back


def create_ferro_top(schema):
    outer_top = Part.makeCylinder(
        schema.magnet_r_outer + schema.ferro_thickness, schema.ferro_thickness
    )
    inner_top = Part.makeCylinder(
        schema.magnet_r_inner,
        schema.ferro_thickness,
    )
    top = outer_top.cut(inner_top)
    top.translate(App.Vector(0, 0, schema.magnet_h + schema.ferro_thickness))
    return top


def create_ferro_side(schema):
    outer_side = Part.makeCylinder(
        schema.magnet_r_outer + schema.ferro_thickness,
        schema.magnet_h,
    )
    inner_side = Part.makeCylinder(
        schema.magnet_r_outer,
        schema.magnet_h,
    )
    side = outer_side.cut(inner_side)
    side.translate(App.Vector(0, 0, schema.ferro_thickness))
    return side


def create_ferro_inside(schema):
    inside = Part.makeCylinder(
        schema.coil_r_inner,
        schema.magnet_h + schema.ferro_thickness,
    )
    inside.translate(App.Vector(0, 0, schema.ferro_thickness))
    return inside


def create_cylinder(schema):
    """Create a cylinder with given radius and height."""
    if schema.cylinder_r_outer and schema.cylinder_h:
        cylinder = Part.makeCylinder(schema.cylinder_r_outer, schema.cylinder_h)
        # Position the cylinder at the correct place (centered)
        # cylinder.translate(App.Vector(0, 0, 1))
        return cylinder
    return None


def create_separate_air_filed(schema):
    """Create a cylinder in crutial air gap and surrounding field in lodspeaker."""
    if (
        schema.magnet_r_inner
        and schema.ferro_thickness
        and schema.air_gap
        and schema.coil_h
    ):
        air_filed_cylinder = Part.makeCylinder(
            schema.magnet_r_inner,
            schema.ferro_thickness
            + schema.magnet_h
            + (schema.coil_h - schema.air_gap) / 2,
        )
        air_filed_cylinder.translate(App.Vector(0, 0, schema.ferro_thickness))
        return air_filed_cylinder
    return None


def merge_ferro_parts(schema):
    # Merge the ferro parts into one single shape if merge_ferro is True
    ferro_back = create_ferro_back(schema)
    ferro_side = create_ferro_side(schema)
    ferro_top = create_ferro_top(schema)

    # Inside is important so i will treat this separately
    # ferro_inside = create_ferro_inside(schema)

    # Combine all ferro parts
    # ferro = ferro_back.fuse(ferro_side).fuse(ferro_top).fuse(ferro_inside)
    ferro = ferro_back.fuse(ferro_side).fuse(ferro_top)

    return ferro.removeSplitter()


def create_loudspeaker(schema):
    doc = App.newDocument("Loudspeaker")
    parts = {
        "membrane": create_membrane(schema),
        "magnet": create_magnet(schema),
        "coil": create_coil(schema),
        "ferro_back": create_ferro_back(schema),
        "ferro_side": create_ferro_side(schema),
        "ferro_top": create_ferro_top(schema),
        "ferro_inside": create_ferro_inside(schema),
    }

    if schema.cylinder_r_outer and schema.cylinder_h:
        cylinder = create_cylinder(schema)
        if cylinder:
            parts["cylinder"] = cylinder

    if schema.split_air_field:
        air_field = create_separate_air_filed(schema)
        if air_field:
            parts["air_field"] = air_field

    if schema.merge_ferro:
        ferro_part = merge_ferro_parts(schema)
        [
            parts.pop(key)
            for key in list(parts.keys())
            if "ferro" in key and key != "ferro_inside"
        ]
        parts["ferro"] = ferro_part

    for name, part in parts.items():
        obj = doc.addObject("Part::Feature", name)
        obj.Shape = part

    doc.recompute()
    return parts


def calculate_cone_field(coil_radius, membrane_radius, membrane_h):
    imagine_h = (
        (membrane_h * membrane_radius) - (membrane_h * coil_radius)
    ) / coil_radius
    cone_side_field = (
        np.pi
        * membrane_radius
        * np.sqrt(((imagine_h + membrane_h) ** 2) + membrane_radius**2)
    ) - (np.pi * coil_radius * np.sqrt(imagine_h**2 * coil_radius**2))
    return cone_side_field


def create_room(loudspeaker_driver, size=(8000, 8000, 4000)):
    """Create a room with specified dimensions (width, length, height)."""
    length = size[0]
    width = size[1]
    height = size[2]
    room = Part.makeBox(length, height, width)
    room.translate(
        App.Vector(
            -(length / 2),
            -((loudspeaker_driver.cylinder_h * 10) / 2),
            loudspeaker_driver.cylinder_h - loudspeaker_driver.membrane_r1 * 4,
        )
    )
    return room


def create_surrounding_box(loudspeaker_driver):
    r = loudspeaker_driver.membrane_r1
    h = loudspeaker_driver.cylinder_h
    width = r * 3
    depth = h * 10
    height = r * 4

    surrounding_box = Part.makeBox(
        width,
        depth,
        height,
    )

    surrounding_box.translate(
        App.Vector(
            -(width / 2), -(depth / 2), -(height - loudspeaker_driver.cylinder_h) - 1
        )
    )
    return surrounding_box


def get_loudspeaker_driver(
    loudspeaker_driver_parameters: dict[str, Any],
    loudspeaker_geometry_parameters: dict[str, Any],
    cone_membrane: bool = True,
) -> LoudspeakerDriver:
    magnet_with_ferro = loudspeaker_geometry_parameters["magnet_with_ferro"][
        "width"
    ]  # [mm]
    driver_height = loudspeaker_geometry_parameters["driver"]["height"]  # [mm]

    air_gap = loudspeaker_driver_parameters["magnet"]["air_gap_height"]["HAG"]  # [mm]
    coil_height = loudspeaker_driver_parameters["voice_coil"]["winding_height"][
        "HVC"
    ]  # [mm]
    coil_diameter = loudspeaker_driver_parameters["voice_coil"]["VC_diameter"][
        "diameter"
    ]  # [mm]
    whole_membrane_diameter = 143  # [mm] , real
    membrane_diameter = loudspeaker_driver_parameters["diaphragm"][
        "effective_diameter"
    ][
        "diameter"
    ]  # [mm]

    # effective membrane radius (137%2)=68.5
    # effective coil radius 38%2=19
    # Temporary H proportion (93÷188) ×70 = 34.627659574

    ferro_thickness = loudspeaker_geometry_parameters["ferromagnetic"]["width"]
    coil_width = loudspeaker_geometry_parameters["voice_coil"]["width"]
    driver_height_without_membrane = loudspeaker_geometry_parameters["driver"][
        "height_wo_diameter_cylinder"
    ]

    # Custom parameters
    magnet_h = driver_height - driver_height_without_membrane - 2 * ferro_thickness
    air_width = air_gap

    # Example with flat membrane (set membrane_is_cone=False)
    loudspeaker_driver = LoudspeakerDriver(
        membrane_r1=membrane_diameter,
        membrane_r2=coil_diameter,
        membrane_h=driver_height - driver_height_without_membrane,
        membrane_is_cone=cone_membrane,
        coil_r_outer=coil_diameter,
        coil_r_inner=coil_diameter - coil_width,
        coil_h=coil_height,
        coil_embed=0.5,  # percentage of magnet_h
        magnet_r_outer=magnet_with_ferro - ferro_thickness,
        magnet_r_inner=coil_diameter + air_width,
        magnet_h=magnet_h,
        ferro_thickness=ferro_thickness,
        air_gap=air_gap,
        split_air_field=True,
        merge_ferro=True,
        cylinder_r_outer=whole_membrane_diameter,
        cylinder_h=driver_height + 2,
    )

    return loudspeaker_driver


def create_loudspeaker_driver(
    loudspeaker_driver: LoudspeakerDriver,
    box: bool = True,
    room: bool = True,
    without_fillers: bool = True,
):

    loudspeaker_parts = create_loudspeaker(loudspeaker_driver)
    base_part = "cylinder"
    parts_to_cut = ["magnet", "ferro", "ferro_inside", "air_field", "coil"]
    cut_parts_from_loudspeaker(loudspeaker_parts, base_part, parts_to_cut)

    if box:
        doc = App.activeDocument()
        surrounding_box = create_surrounding_box(loudspeaker_driver)
        box_obj = doc.addObject("Part::Feature", "box")
        box_obj.Shape = surrounding_box

        cut_object = doc.addObject("Part::Feature", "box_cut")
        cut_object.Shape = box_obj.Shape.cut(loudspeaker_parts["cylinder"])
        doc.recompute()
    if room:
        doc = App.activeDocument()
        room = create_room(loudspeaker_driver)
        room_obj = doc.addObject("Part::Feature", "room")
        room_obj.Shape = room
        room_cut = doc.addObject("Part::Feature", "room_cut")
        room_cut.Shape = room_obj.Shape.cut(surrounding_box)
        room_cut.Shape = room_cut.Shape.cut(loudspeaker_parts["cylinder"])
        doc.recompute()

    if without_fillers:
        remove_parts(
            [
                "air_field",
                "box",
                # "cylinder",
                "cylinder_cut",
                "room",
            ]
        )


if __name__ == "__main__":

    # This script is strange because u should run it in FreeCad python console using command:
    # exec(open("/absolute/path/to/loudspeaker-solver/boomspeaver/loudspeaker/geometry/create_driver_prv6MB400.py").read())

    # https://loudspeakerdatabase.com/PRV/6MB400
    input_parameter_path = "absolute/path/to/examples/loudspeaker-solver/examples/prv_audio_6MB400_8ohm.json"
    input_parameter_path = "/home/freetzz/repo/ls_wip/loudspeaker-solver/examples/prv_audio_6MB400_8ohm.json"
    loudspeaker_driver_parameters = load_json_file(input_parameter_path)

    input_ls_geometry_path = "absolute/path/to/examples/loudspeaker-solver/examples/prv_audio_6MB400_8ohm_geometry.json"
    input_ls_geometry_path = "/home/freetzz/repo/ls_wip/loudspeaker-solver/examples/prv_audio_6MB400_8ohm_geometry.json"
    loudspeaker_geometry_parameters = load_json_file(input_ls_geometry_path)

    input_room_geometry_path = "absolute/path/to/examples/loudspeaker-solver/examples/prv_audio_6MB400_8ohm_geometry.json"
    input_room_geometry_path = (
        "/home/freetzz/repo/ls_wip/loudspeaker-solver/examples/room_geometry.json"
    )
    room_geometry_parameters = load_json_file(input_room_geometry_path)

    # Setup parameters
    cone_membrane = True
    room = True
    box = True

    loudspeaker_driver = get_loudspeaker_driver(
        loudspeaker_driver_parameters=loudspeaker_driver_parameters,
        loudspeaker_geometry_parameters=loudspeaker_geometry_parameters,
        cone_membrane=cone_membrane,
    )
    create_loudspeaker_driver(
        loudspeaker_driver=loudspeaker_driver,
        box=box,
        room=room,
    )
