# pylint: disable=invalid-name

import FreeCAD as App
import numpy as np
import Part


class LoudspeakerSchema:
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
        membrane_is_cone=True,  # Add this flag to choose between cone or flat
        merge_ferro=False,  # Add a flag to merge ferro parts
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
        self.membrane_is_cone = membrane_is_cone
        self.merge_ferro = merge_ferro


def create_membrane(schema):
    if schema.membrane_is_cone:
        # Create membrane as a cone
        membrane = Part.makeCone(
            schema.membrane_r1, schema.membrane_r2, schema.membrane_h
        )
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


def merge_ferro_parts(schema):
    # Merge the ferro parts into one single shape if merge_ferro is True
    ferro_back = create_ferro_back(schema)
    ferro_side = create_ferro_side(schema)
    ferro_top = create_ferro_top(schema)
    ferro_inside = create_ferro_inside(schema)

    # Combine all ferro parts
    ferro = ferro_back.fuse(ferro_side).fuse(ferro_top).fuse(ferro_inside)
    return ferro


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


# TODO to check
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


# https://loudspeakerdatabase.com/PRV/6MB400
magnet_with_ferro = 115  # [mm]
driver_height = 70  # [mm]
air_gap = 8  # [mm]
coil_height = 12  # [mm]
coil_diameter = 38  # [mm]
driver_height = 70  # [mm]
# membrane_diameter = 143  # [mm] , real
membrane_diameter = 137  # [mm] , effective

# effective membrane radius (137%2)=68.5
# effective coil radius 38%2=19
# Temporary H proportion (93÷188) ×70 = 34.627659574

# custom parameters
ferro_thickness = air_gap
air_width = air_gap
coil_width = 1
driver_height_without_membrane = 35

magnet_h = driver_height - driver_height_without_membrane - 2 * ferro_thickness
# Example with flat membrane (set membrane_is_cone=False)
schema = LoudspeakerSchema(
    membrane_r1=membrane_diameter,
    membrane_r2=coil_diameter,
    membrane_h=driver_height - driver_height_without_membrane,  # Corrected expression
    membrane_is_cone=False,  # Change this flag to False to use flat membrane
    coil_r_outer=coil_diameter,
    coil_r_inner=coil_diameter - coil_width,
    coil_h=coil_height,
    coil_embed=0.5,  # percentage of magnet_h
    magnet_r_outer=magnet_with_ferro - ferro_thickness,
    magnet_r_inner=coil_diameter + air_width,
    magnet_h=magnet_h,
    ferro_thickness=ferro_thickness,
    air_gap=air_gap,
    merge_ferro=True,
)

create_loudspeaker(schema)

# exec(open("/home/freetzz/repo/fenics/geometric/loudspeaker_driver/create_driver_prv6MB400.py").read())
