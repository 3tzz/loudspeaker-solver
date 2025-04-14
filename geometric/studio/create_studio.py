import FreeCAD as App
import Part

# Nowy dokument
doc = App.newDocument("Studio")

# Wymiary pokoju (mm)
length = 3700
width = 5200
height = 2700

# Tworzenie bryły pokoju
room = Part.makeBox(length, width, height)

# Dodanie do FreeCAD GUI
obj = doc.addObject("Part::Feature", "Room")
obj.Shape = room

# Przeliczenie modelu
doc.recompute()

# exec(open("/home/freetzz/repo/fenics/geometric/studio/create_studio.py").read())
