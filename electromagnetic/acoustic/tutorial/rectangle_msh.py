import sys

import gmsh

# Initialize Gmsh
gmsh.initialize()
gmsh.model.add("rectangle")

lc = 1e-2
gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
gmsh.model.geo.addPoint(0.1, 0, 0, lc, 2)
gmsh.model.geo.addPoint(0.1, 0.1, 0, lc, 3)
p4 = gmsh.model.geo.addPoint(0, 0.1, 0, lc)

gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(3, 2, 2)
gmsh.model.geo.addLine(3, p4, 3)
gmsh.model.geo.addLine(4, 1, p4)

gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.005)  # Maximum element size
gmsh.option.setNumber(
    "Mesh.CharacteristicLengthMin", 0.005
)  # Minimum element size (same for uniform size)

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(1, [1, 2, 4], 5)
gmsh.model.addPhysicalGroup(2, [1], name="Rectangle")

gmsh.model.mesh.generate(2)

gmsh.write("rectangle.msh")

if "-nopopup" not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()
