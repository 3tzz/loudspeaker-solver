import FreeCAD as App
import Part


def load_step_file(file_path):
    """Wczytuje plik STEP i zwraca obiekt Part."""
    return Part.read(file_path)


def place_parts_in_center(room_obj, magnet_obj):
    """Umieszcza czesci na środku."""


def cut_parts_from_room(room_obj, magnet_obj):
    """Odejmuje czesci od cyindra."""


def save_result(room_obj, magnet_obj, output_path):
    """Zapisuje wynikowy kształt do pliku STEP, zawierający zarówno magnes, jak i pokój."""
    # Tworzenie nowego dokumentu i dodanie obiektów
    doc = App.newDocument("MagnetInRoom")

    # Dodanie pokoju (powietrze) jako obiekt
    room_part = doc.addObject("Part::Feature", "Room_Air")
    room_part.Shape = room_obj  # Room with air

    # Dodanie magnesu jako osobnego obiektu
    magnet_part = doc.addObject("Part::Feature", "Magnet")
    magnet_part.Shape = magnet_obj  # Magnet body

    # Przeliczenie modelu
    doc.recompute()

    # Zapisz wynik do STEP (osobno dla pokoju i magnesu w tym samym pliku)
    doc.exportStep(output_path)


def main():
    # Ścieżki do plików STEP
    cylinder_step_path = (
        "/home/freetzz/repo/fenics/geometric/driver_cylinder/export/Cylinder.brep"
    )
    magnet_step_path = (
        "/home/freetzz/repo/fenics/geometric/loudspeaker_driver/export/magnet.brep"
    )
    ferro_inside_step_path = "/home/freetzz/repo/fenics/geometric/loudspeaker_driver/export/ferro_inside.brep"
    ferro_step_path = (
        "/home/freetzz/repo/fenics/geometric/loudspeaker_driver/export/ferro.brep"
    )

    # Wczytanie plików STEP
    cylinder_obj = load_step_file(cylinder_step_path)
    magnet_obj = load_step_file(magnet_step_path)
    ferro_inside_obj = load_step_file(ferro_inside_step_path)
    ferro_obj = load_step_file(ferro_step_path)

    # Umieszczanie magnesu w środku pokoju
    placed_parts = place_parts_in_center(
        cylinder_obj, magnet_obj, ferro_obj, ferro_inside_obj
    )

    # Odejście magnesu od pokoju, otrzymujemy oba obiekty (pokój i magnes)
    cylinder_obj, parts_cut = cut_parts_from_room(cylinder_obj_obj, placed_parts)

    # Zapisz wynik do STEP (osobno dla pokoju i magnesu w tym samym pliku)
    save_result()


# Uruchomienie głównej funkcji
if __name__ == "__main__":
    main()
    # exec(open("/home/freetzz/repo/fenics/geometric/tools/magnet_test.py").read())
