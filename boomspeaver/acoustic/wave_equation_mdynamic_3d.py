# pylint: disable=invalid-name
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-arguments
# pylint: disable=redefined-outer-name

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from boomspeaver.acoustic.microphone_pressure import MicrophonePressure
from boomspeaver.tools.data import (
    get_repo_dir,
    load_numpy_file,
    pad_vector,
    save_numpy_file,
)
from boomspeaver.tools.dsp.dsp import get_resolution
from boomspeaver.tools.fem import (
    close_xdmf,
    convert_to_form,
    get_surface_cords,
    get_surface_edge_coords,
    initialize_fem_function,
    initialize_xdmf,
    interpolate_surface_values,
    load_mesh,
    precalculate_surface_dofs,
    set_bcs,
    wave_equation,
)


def time_loop_wave(
    domain: mesh.Mesh,
    bilinear_form: ufl.form.Form,
    linear_form: ufl.form.Form,
    time: np.ndarray,
    membrane_displacement: np.ndarray,
    u: fem.function.Function,
    u0: fem.function.Function,
    u1: fem.function.Function,
    force_function: fem.function.Function,
    bc: Any,
    xdmf: io.utils.XDMFFile,
    microphone: MicrophonePressure | None,
    target_dofs: Any,
) -> tuple[io.utils.XDMFFile, np.ndarray]:
    """Time step loop simulation."""

    p_mic = np.zeros_like(time)

    force_function.x.array[:] = 0.0
    u.x.array[:] = 0.0
    u1.x.array[:] = 0.0
    u0.x.array[:] = 0.0

    for idx, t in enumerate(time):
        print(f"Index: {idx}, Time: {t}.")

        if idx < membrane_displacement.shape[0]:
            displacement = interpolate_surface_values(
                membrane_displacement=membrane_displacement[idx, :, :],
                x_target=target_dofs[1],
                y_target=target_dofs[2],
            )
            u1.x.array[target_dofs[0]] = displacement

        if bc is not None:
            A = fem.petsc.assemble_matrix(bilinear_form, bcs=bc)
        else:
            A = fem.petsc.assemble_matrix(bilinear_form)
        A.assemble()
        b = fem.petsc.assemble_vector(linear_form)

        solver = PETSc.KSP().create(domain.comm)
        solver.setOperators(A)
        solver.setType("cg")
        solver.getPC().setType("ilu")
        solver.solve(b, u.x.petsc_vec)
        if microphone is not None:
            p_f = microphone.listen(u)
            p_f = domain.comm.gather(p_f, root=0)
            if domain.comm.rank == 0:
                assert p_f is not None
                p_mic[idx] = np.hstack(p_f)
        u0.x.array[:] = u1.x.array
        u1.x.array[:] = u.x.array
        xdmf.write_function(u, t)
    return xdmf, p_mic


def simulate_wave(
    domain: mesh.Mesh,
    facet_tags: mesh.MeshTags,
    time: np.ndarray,
    membrane_displacement: np.ndarray,
    output_path: Path,
    mic_coordinates: bool,
    membrane_surface: int,
    bcs: str = "dirichlet",
    bcs_surfaces: list[int] | None = None,
):
    """Setup and Simulate wave equation."""

    V = fem.functionspace(domain, ("Lagrange", 1))

    u1 = initialize_fem_function(V, "u1")
    u0 = initialize_fem_function(V, "u0")
    u = initialize_fem_function(V, "u")
    f = initialize_fem_function(V, "f")

    target_dofs = precalculate_surface_dofs(
        V=V, domain=domain, facet_tags=facet_tags, surface_tag=membrane_surface
    )

    bc = set_bcs(bcs, domain, V, facet_tags=facet_tags, surface_tags=bcs_surfaces)

    xdmf = initialize_xdmf(domain=domain, output_path=output_path)

    dt = get_resolution(time)
    a, l = wave_equation(
        V=V,
        domain=domain,
        facet_tags=facet_tags,
        u0=u0,
        u1=u1,
        dt=dt,
        force=f,
        surface_force_id=membrane_surface,
    )
    bilinear_form, linear_form = convert_to_form(input_forms=[a, l])

    if mic_coordinates:
        force_surface_dofs = get_surface_cords(
            domain=domain, V=V, facet_tags=facet_tags, tag=membrane_surface
        )
        mic_coordinates = np.mean(force_surface_dofs, axis=0)
        mic_coordinates[1] += 1
        print("Microphone coordinate:", mic_coordinates)
        microphone = MicrophonePressure(domain, np.array(mic_coordinates))
    else:
        microphone = None

    xdmf, p_mic = time_loop_wave(
        domain=domain,
        bilinear_form=bilinear_form,
        linear_form=linear_form,
        time=time,
        membrane_displacement=membrane_displacement,
        u=u,
        u0=u0,
        u1=u1,
        force_function=f,
        bc=bc,
        xdmf=xdmf,
        microphone=microphone,
        target_dofs=target_dofs,
    )
    save_numpy_file(output_path=output_path.with_suffix(".npy"), data=p_mic)
    close_xdmf(xdmf)


def main(
    time_input_path: Path,
    membrane_input_path: Path,
    mesh_path: Path,
    output_path: Path,
    mic_coordinates: bool,
    membrane_surface: list[int] = [2],
    bcs_surfaces: list[int] | None = None,
):
    """Load input data and simulate."""
    assert isinstance(time_input_path, Path)
    assert isinstance(membrane_input_path, Path)
    assert isinstance(mesh_path, Path)
    assert isinstance(output_path, Path)
    assert isinstance(mic_coordinates, bool)
    assert isinstance(membrane_surface, int)
    assert isinstance(bcs_surfaces, list) or bcs_surfaces is None

    time = load_numpy_file(time_input_path)
    assert len(time.shape) == 1

    membrane_displacement = load_numpy_file(membrane_input_path)
    assert len(membrane_displacement.shape) == 3
    assert membrane_displacement.shape[-1] == 3

    domain, _, facet_tags = load_mesh(mesh_path, dim=3, return_all=True)
    available_ids = set(facet_tags.values)

    assert membrane_surface in available_ids

    if bcs_surfaces:
        assert all(surface in available_ids for surface in bcs_surfaces)

    simulate_wave(
        domain=domain,
        facet_tags=facet_tags,
        time=time,
        membrane_displacement=membrane_displacement,
        output_path=output_path,
        mic_coordinates=mic_coordinates,
        membrane_surface=membrane_surface,
        bcs_surfaces=bcs_surfaces,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate membrane motion based on input signal and loudspeaker parameters."
    )
    parser.add_argument(
        "--time_input_path",
        type=str,
        default="examples/time_vector_20ms_48kHz.npy",
        help="Path to the NumPy file containing the time vector.",
    )
    parser.add_argument(
        "--membrane_input_path",
        type=str,
        required=True,
        help="Path to the NumPy file containing the membrane displacement along time. Generated from boomspeaver/structural/the_membrane.py",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="examples/loudspeaker_in_room.msh",
        help="Path to 3D room mesh.",
    )
    parser.add_argument(
        "--membrane_ids",
        type=int,
        default=2,
        help="Surface id(s) for membrane input from structural part.",
    )
    parser.add_argument(
        "--bcs_ids",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7],
        help="Surface id(s) Dirichlet boundary conditions.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/wave_equation_dynamic.xdmf",
        help="Path to the output XDMF file.",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Listen along membrane radius using MicrophonePressure class.",
    )

    args = parser.parse_args()

    main(
        time_input_path=Path(args.time_input_path),
        membrane_input_path=Path(args.membrane_input_path),
        mesh_path=Path(args.mesh_path),
        membrane_surface=args.membrane_ids,
        bcs_surfaces=args.bcs_ids,
        output_path=Path(args.output_path),
        mic_coordinates=args.listen,
    )
