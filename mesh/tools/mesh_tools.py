import sys

import gmsh


def load_gmsh(
    input_path: str,
    name: str,
    metrics: None | str,
    mesh_size_max: None | float,
    mesh_size_min: None | float,
) -> None:
    """Initializes GMSH with the given input file."""
    gmsh.initialize(sys.argv)
    gmsh.model.add(name)
    if metrics:
        gmsh.option.setString("Geometry.OCCTargetUnit", metrics)
    gmsh.merge(input_path)
    if mesh_size_max:
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    if mesh_size_min:
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)


def configure_mesh(mesh_size_max: float, mesh_size_min: float) -> None:
    """Configures the mesh settings."""
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size_max)
    gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size_min)


def generate_mesh(output_path: str) -> None:
    """Generates and saves the mesh."""
    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)


def define_new_volume(surfaces: list[int]):
    """
    Define new volume according to provided surfaces
    (make sure that they are in proper order,
    best way is to add this volume in GMSH GUI and check in geo file order).
    """
    surface_loop = gmsh.model.occ.addSurfaceLoop(
        surfaces
    )  # i add this in gmsh gui and check order
    _ = gmsh.model.occ.addVolume([surface_loop])
    gmsh.model.occ.synchronize()


def define_physical_groups(
    volumes: list[int], surfaces_groups: dict[str, list[int]], name: str
) -> None:
    """Defines physical groups for the mesh."""
    group_id = 1
    for group_name, surfaces in surfaces_groups.items():
        gmsh.model.addPhysicalGroup(2, surfaces, group_id, group_name)
        group_id += 1
    for body_name, volume in volumes.items():
        gmsh.model.addPhysicalGroup(3, volume, group_id, body_name)
        group_id += 1


def set_surface_mesh_size(
    mesh_size: float, field_id: int, default_mesh_size: float, surfaces: list[int]
) -> None:
    """Sets mesh size for specified surfaces."""
    gmsh.model.mesh.field.add("Constant", field_id)
    gmsh.model.mesh.field.setNumbers(field_id, "SurfacesList", surfaces)
    gmsh.model.mesh.field.setNumber(field_id, "VIn", mesh_size)
    gmsh.model.mesh.field.setNumber(field_id, "VOut", default_mesh_size)


def apply_mesh_sizing(
    mesh_size: float,
    default_mesh_size: float,
    surfaces_groups: dict[str, list[int]],
) -> list[int]:
    """Applies mesh size settings to different surface groups."""
    fields = []
    field_id = 1
    for _, surfaces in surfaces_groups.items():
        set_surface_mesh_size(mesh_size, field_id, default_mesh_size, surfaces)
        fields.append(field_id)
        field_id += 1
    gmsh.model.mesh.field.add("Min", field_id)
    gmsh.model.mesh.field.setNumbers(field_id, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_id)


def assign_mesh_size(surfaces_groups: dict[str, list[int]], mesh_size: float):
    return {key: mesh_size for key in surfaces_groups.keys()}


def update_mesh_size_for_keys(
    mesh_sizes: dict[str, float], names_to_update: set[str], new_size: float
):
    for name in names_to_update:
        if name in mesh_sizes:
            mesh_sizes[name] = new_size
    return mesh_sizes


def apply_mshs_sizing(
    mesh_size_dict: dict[str, float],
    default_mesh_size: float,
    surfaces_groups: dict[str, list[int]],
) -> list[int]:
    """Applies mesh size settings to different surface groups."""
    fields = []
    field_id = 1
    for name, surfaces in surfaces_groups.items():
        if name not in mesh_size_dict.keys():
            raise ValueError
        mesh_size = mesh_size_dict[name]
        if mesh_size == default_mesh_size:
            continue
        set_surface_mesh_size(mesh_size, field_id, default_mesh_size, surfaces)
        fields.append(field_id)
        field_id += 1
    gmsh.model.mesh.field.add("Min", field_id)
    gmsh.model.mesh.field.setNumbers(field_id, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_id)


def merge_surface_groups(surfaces_groups, duplicate_surfaces_map):
    return {
        key: value + [k for k, v in duplicate_surfaces_map.items() if v in value]
        for key, value in surfaces_groups.items()
    }


def get_from_dict(input_dict, keys_set):
    return {key: input_dict[key] for key in keys_set if key in input_dict}


def check_volumes_and_surfaces() -> None:
    print(gmsh.model.getEntities(dim=3))
    print(gmsh.model.getEntities(dim=2))
    raise
