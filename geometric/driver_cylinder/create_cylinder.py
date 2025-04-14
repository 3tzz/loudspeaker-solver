import FreeCAD as App
import Part

# Nowy dokument
doc = App.newDocument("Cylinder")

# Wymiary
driver_height = 70  # [mm]
membrane_diameter = 143  # [mm]

# Tworzenie cylindra (wysokość, promień)
radius = membrane_diameter / 2  # Promień z średnicy
cylinder = Part.makeCylinder(radius, driver_height)

# Dodanie do FreeCAD GUI
obj = doc.addObject("Part::Feature", "Cylinder")
obj.Shape = cylinder

# Przeliczenie modelu
doc.recompute()

# exec(open("/home/freetzz/repo/fenics/geometric/driver_cylinder/create_cylinder.py").read())
