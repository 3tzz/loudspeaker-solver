import FreeCAD as App
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
        magnet_r_outer,
        magnet_r_inner,
        magnet_h,
    ):
        self.membrane_r1 = membrane_r1
        self.membrane_r2 = membrane_r2
        self.membrane_h = membrane_h
        self.coil_r_outer = coil_r_outer
        self.coil_r_inner = coil_r_inner
        self.coil_h = coil_h
        self.magnet_r_outer = magnet_r_outer
        self.magnet_r_inner = magnet_r_inner
        self.magnet_h = magnet_h


def create_membrane(schema):
    membrane = Part.makeCone(schema.membrane_r1, schema.membrane_r2, schema.membrane_h)
    membrane.translate(App.Vector(0, 0, schema.membrane_h / 2))
    return membrane


def create_coil(schema):
    outer_cylinder = Part.makeCylinder(schema.coil_r_outer, schema.coil_h)
    inner_cylinder = Part.makeCylinder(schema.coil_r_inner, schema.coil_h)
    coil = outer_cylinder.cut(inner_cylinder)
    coil.translate(App.Vector(0, 0, schema.magnet_h / 2))
    return coil


def create_magnet(schema):
    outer_magnet = Part.makeCylinder(schema.magnet_r_outer, schema.magnet_h)
    inner_magnet = Part.makeCylinder(schema.magnet_r_inner, schema.magnet_h)
    magnet = outer_magnet.cut(inner_magnet)
    return magnet


def create_loudspeaker(schema):
    doc = App.newDocument("Loudspeaker")
    parts = {
        "membrane": create_membrane(schema),
        "magnet": create_magnet(schema),
        "coil": create_coil(schema),
    }

    for name, part in parts.items():
        obj = doc.addObject("Part::Feature", name)
        obj.Shape = part

    doc.recompute()


schema = LoudspeakerSchema(
    membrane_r1=30,
    membrane_r2=80,
    membrane_h=40,
    coil_r_outer=10,
    coil_r_inner=8,
    coil_h=15,
    magnet_r_outer=20,
    magnet_r_inner=12,
    magnet_h=10,
)

create_loudspeaker(schema)
