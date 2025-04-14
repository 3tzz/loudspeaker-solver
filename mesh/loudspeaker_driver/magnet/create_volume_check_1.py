import gmsh

# Initialize GMSH
gmsh.initialize()

# Define geometry parameters
lc = 1e-2  # Characteristic length for mesh
L = 10.0  # Outer cylinder height
R_outer = 5.0  # Outer radius
R_inner = 3.0  # Inner radius (cavity)
thickness = 1.0  # Thickness of the outer shell

# Define points for the outer surface (top and bottom)
p0 = gmsh.model.occ.addPoint(0, 0, 0, lc)  # Bottom center
p1 = gmsh.model.occ.addPoint(R_outer, 0, 0, lc)  # Outer radius point on x-axis
p2 = gmsh.model.occ.addPoint(0, R_outer, 0, lc)  # Outer radius point on y-axis
p3 = gmsh.model.occ.addPoint(0, 0, L, lc)  # Top center
p4 = gmsh.model.occ.addPoint(R_outer, 0, L, lc)  # Top point on x-axis

# Create lines for the outer surface
line_0_1 = gmsh.model.occ.addLine(p0, p1)  # Bottom face lines
line_1_2 = gmsh.model.occ.addLine(p1, p2)
line_2_0 = gmsh.model.occ.addLine(p2, p0)

line_3_4 = gmsh.model.occ.addLine(p3, p4)  # Top face lines
line_4_1 = gmsh.model.occ.addLine(p4, p1)
line_1_3 = gmsh.model.occ.addLine(p1, p3)

# Create the outer curve loop and surface
loop_outer = gmsh.model.occ.addCurveLoop(
    [line_0_1, line_1_2, line_2_0, line_3_4, line_4_1, line_1_3]
)
surface_outer = gmsh.model.occ.addPlaneSurface([loop_outer])

# Now, let's add the inner hole (a cylinder)
p5 = gmsh.model.occ.addPoint(R_inner, 0, 0, lc)  # Inner radius bottom
p6 = gmsh.model.occ.addPoint(0, R_inner, 0, lc)  # Inner radius bottom (other direction)

# Create lines for the inner surface
line_5_6 = gmsh.model.occ.addLine(p5, p6)

# Create a surface for the hole, use a similar method as before
loop_inner = gmsh.model.occ.addCurveLoop([line_5_6])
surface_inner = gmsh.model.occ.addPlaneSurface([loop_inner])

# Now create a Boolean difference to subtract the inner surface from the outer
gmsh.model.occ.cut([3], [2], [])

# Synchronize the model and generate the mesh
gmsh.model.occ.synchronize()

# Finalize GMSH

gmsh.fltk.run()

# Finalize GMSH
gmsh.finalize()
