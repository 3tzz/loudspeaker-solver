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
    force: np.ndarray,
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

    for idx, (t, fv) in enumerate(zip(time, force)):
        print(f"Index: {idx}, Time: {t}, Force value: {fv}.")

        force_function.x.array[target_dofs[0]] = fv

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
    force: np.ndarray,
    output_path: Path,
    mic_coordinates: bool,
    force_surface: list[int],
    bcs: str = "neumann",
    bcs_surfaces: list[int] | None = None,
):
    """Setup and Simulate wave equation."""

    V = fem.functionspace(domain, ("Lagrange", 1))

    u1 = initialize_fem_function(V, "u1")
    u0 = initialize_fem_function(V, "u0")
    u = initialize_fem_function(V, "u")
    f = initialize_fem_function(V, "f")

    target_dofs = precalculate_surface_dofs(
        V=V, domain=domain, facet_tags=facet_tags, surface_tag=force_surface
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
        surface_force_id=force_surface,
    )
    bilinear_form, linear_form = convert_to_form(input_forms=[a, l])

    if mic_coordinates:
        force_surface_dofs = get_surface_cords(
            domain=domain, V=V, facet_tags=facet_tags, tag=force_surface
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
        force=force,
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
    signal_input_path: Path,
    mesh_path: Path,
    output_path: Path,
    mic_coordinates: bool,
    force_surface: list[int] = [2],
    bcs_surfaces: list[int] | None = None,
):
    """Load input data and simulate."""
    assert isinstance(time_input_path, Path)
    assert isinstance(signal_input_path, Path)
    assert isinstance(mesh_path, Path)
    assert isinstance(output_path, Path)
    assert isinstance(mic_coordinates, bool)
    assert isinstance(force_surface, int)
    assert isinstance(bcs_surfaces, list) or bcs_surfaces is None

    time = load_numpy_file(time_input_path)
    assert len(time.shape) == 1

    force = load_numpy_file(signal_input_path)
    assert len(force.shape) == 1

    if len(force) > len(time):
        # force = force[:len(time)]
        print("Cut force vector to time length.")
        force = force[len(force) - len(time) :]
    else:
        force = pad_vector(force, time)

    domain, _, facet_tags = load_mesh(mesh_path, dim=3, return_all=True)
    available_ids = set(facet_tags.values)

    assert force_surface in available_ids

    if bcs_surfaces:
        assert all(surface in available_ids for surface in bcs_surfaces)

    simulate_wave(
        domain=domain,
        facet_tags=facet_tags,
        time=time,
        force=force,
        output_path=output_path,
        mic_coordinates=mic_coordinates,
        force_surface=force_surface,
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
        "--signal_input_path",
        type=str,
        required=True,
        help="Path to the NumPy file containing the input signal. Assigned in the centre of geometry.",
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default="examples/loudspeaker_in_room.msh",
        help="Path to 3D room mesh.",
    )
    parser.add_argument(
        "--force_ids",
        type=int,
        default=2,
        help="Surface id(s) for signal input force.",
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
        default="output/wave_equation_piston.xdmf",
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
        signal_input_path=Path(args.signal_input_path),
        mesh_path=Path(args.mesh_path),
        force_surface=args.force_ids,
        bcs_surfaces=args.bcs_ids,
        output_path=Path(args.output_path),
        mic_coordinates=args.listen,
    )
