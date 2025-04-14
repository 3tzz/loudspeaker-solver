import FreeCAD as App
import Part


def load_step_file(file_path):
    """Wczytuje plik STEP i zwraca obiekt Part."""
    return Part.read(file_path)


def place_magnet_in_center(room_obj, magnet_obj):
    """Umieszcza magnes na środku pokoju."""
    room_bbox = room_obj.BoundBox
    room_center = room_bbox.Center

    # Pobranie wymiarów magnesu
    magnet_bbox = magnet_obj.BoundBox
    magnet_center = magnet_bbox.Center

    # Obliczenie przesunięcia magnesu na środek pokoju
    dx = room_center.x - magnet_center.x
    dy = room_center.y - magnet_center.y
    dz = room_center.z - magnet_center.z

    # Przemieszczanie magnesu do środka pokoju
    magnet_obj.translate(App.Vector(dx, dy, dz))

    return magnet_obj


def cut_magnet_from_room(room_obj, magnet_obj):
    """Odejmuje magnes od pokoju."""
    return room_obj.cut(magnet_obj)


def save_result(room_cut, output_path):
    """Zapisuje wynikowy kształt do pliku STEP."""
    room_cut.exportStep(output_path)


def main():
    # Ścieżki do plików STEP
    room_step_path = "/home/freetzz/repo/fenics/geometric/studio/export/studio.step"
    magnet_step_path = (
        "/home/freetzz/repo/fenics/geometric/loudspeaker_driver/export/ls-magnet.step"
    )

    # Wczytanie plików STEP
    room_obj = load_step_file(room_step_path)
    magnet_obj = load_step_file(magnet_step_path)

    # Umieszczanie magnesu w środku pokoju
    placed_magnet = place_magnet_in_center(room_obj, magnet_obj)

    # Odejście magnesu od pokoju
    room_cut = cut_magnet_from_room(room_obj, placed_magnet)

    # Tworzenie nowego dokumentu i dodanie wynikowego kształtu
    doc = App.newDocument("MagnetInRoom")
    room_part = doc.addObject("Part::Feature", "Room_Air")
    room_part.Shape = room_cut

    # Dodanie magnesu jako osobnego obiektu
    magnet_part = doc.addObject("Part::Feature", "Magnet")
    magnet_part.Shape = magnet_cut  # Magnet body

    # Przeliczenie modelu
    doc.recompute()

    # Zapisz wynik do STEP
    save_result(
        room_cut, "/home/freetzz/repo/fenics/geometric/tools/export/magnet_in_room.step"
    )


# Uruchomienie głównej funkcji
if __name__ == "__main__":
    main()

    # exec(open("/home/freetzz/repo/fenics/geometric/tools/magnet_cutter.py").read())
