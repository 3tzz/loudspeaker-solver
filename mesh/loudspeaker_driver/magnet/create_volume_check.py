import gmsh

# Initialize GMSH
gmsh.initialize()

# Set the mesh size
lc = 0.1
L = 1.0  # Length of the box
H = 0.5  # Height of the box

# Define points
p0 = gmsh.model.occ.addPoint(0, 0, 0, lc)  # Bottom-left corner
p1 = gmsh.model.occ.addPoint(L, 0, 0, lc)  # Bottom-right corner
p2 = gmsh.model.occ.addPoint(L, H, 0, lc)  # Top-right corner
p3 = gmsh.model.occ.addPoint(0, H, 0, lc)  # Top-left corner

# Define lines between the points (edges of the box)
l1 = gmsh.model.occ.addLine(p0, p1)  # Line from p0 to p1
l2 = gmsh.model.occ.addLine(p1, p2)  # Line from p1 to p2
l3 = gmsh.model.occ.addLine(p2, p3)  # Line from p2 to p3
l4 = gmsh.model.occ.addLine(p3, p0)  # Line from p3 to p0

# Create a curve loop from the defined lines
# loop = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
loop = gmsh.model.occ.addCurveLoop([l1, l3, l4, l2])

# Create a surface from the curve loop
surface = gmsh.model.occ.addPlaneSurface([loop])

# Synchronize the model to ensure changes are applied
gmsh.model.occ.synchronize()

# Generate the mesh (in 2D for surface)
gmsh.model.mesh.generate(2)

# Launch GMSH GUI to visualize the geometry and the mesh
gmsh.fltk.run()

# Finalize GMSH
gmsh.finalize()
