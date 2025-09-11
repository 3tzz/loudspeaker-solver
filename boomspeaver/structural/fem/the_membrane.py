# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from boomspeaver.acoustic.microphone_pressure import MicrophonePressure
from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.data import (
    get_repo_dir,
    load_numpy_file,
    pad_vector,
    save_numpy_file,
)
from boomspeaver.tools.dsp.dsp import get_resolution
from boomspeaver.tools.fem import (
    Shape,
    close_xdmf,
    compute_polar_coordinates,
    convert_to_form,
    create_annulus_mesh,
    create_circular_mesh,
    initialize_fem_function,
    initialize_xdmf,
    set_bcs,
    wave_equation,
)


def validate(
    loudspeaker_params_path: Path,
):
    """Validate input sctipt data."""
    assert isinstance(loudspeaker_params_path, Path)
    assert loudspeaker_params_path.exists
    assert loudspeaker_params_path.suffix == ".json"


def load_loudspeaker_parameters(
    loudspeaker_params_path: Path,
) -> tuple[float, float]:
    """Load needed parameters from loudspeaker config."""
    validate(loudspeaker_params_path)
    loudspeaker_params = Loudspeaker.from_json(input_path=loudspeaker_params_path)
    loudspeaker_params.print_main_params()

    r_coil = float(loudspeaker_params.voice_coil.VC_diameter) / 2
    r_diaphragm = float(loudspeaker_params.diaphragm.diameter) / 2
    rho = float(loudspeaker_params.diaphragm.rho)
    youngs_modulus = float(loudspeaker_params.diaphragm.E)
    rms = float(loudspeaker_params.suspensions.rms)
    highest_frequency = float(loudspeaker_params.frequency_response["max"])
    return r_diaphragm, r_coil, rho, youngs_modulus, rms, highest_frequency


def time_loop_wave(
    domain: mesh.Mesh,
    bilinear_form: fem.forms.Form,
    linear_form: fem.forms.Form,
    time: np.ndarray,
    force: np.ndarray,
    force_shape: Shape | None,
    u: fem.function.Function,
    u0: fem.function.Function,
    u1: fem.function.Function,
    force_function: fem.function.Function,
    bc: fem.bcs.DirichletBC,
    xdmf: io.utils.XDMFFile,
    microphone: MicrophonePressure | None,
) -> tuple[io.utils.XDMFFile, np.ndarray, np.ndarray]:
    """Time step loop simulation."""
    p_mic = None
    p_mic_list = []
    membrane = []

    target_dofs = fem.locate_dofs_geometrical(
        force_function.function_space, lambda x: force_shape.shape(x)
    )
    for idx, (t, fv) in enumerate(zip(time, force)):
        print(f"Index: {idx}, Time: {t}, Force value: {fv}.")
        if force_shape is None:
            force_function.interpolate(lambda x: np.full(x.shape[1], fv))
        else:
            assert isinstance(force_shape, Shape)
            u1.x.array[target_dofs] = fv

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
                p_mic_list.append(np.hstack(p_f))
        u0.x.array[:] = u1.x.array
        u1.x.array[:] = u.x.array
        xdmf.write_function(u, t)
        u_array = u.x.array.copy()
        membrane.append(u_array)
    if microphone:
        p_mic = np.stack(p_mic_list, axis=0).squeeze(-1)
    return xdmf, p_mic, np.array(membrane)


def simulate_wave(
    domain: mesh.Mesh,
    time: np.ndarray,
    force: np.ndarray | None,
    initial_condition: Shape | None,
    output_path: Path,
    shape_profile: Shape | None,
    c: float,
    mic_coordinates: list[float] | None,
    save_membrane: bool,
    damping: float | None = None,
    bcs: str = "neumann",
) -> None:
    """Setup and simulate wave equation."""
    V = fem.functionspace(domain, ("Lagrange", 1))

    u1 = initialize_fem_function(V, "u1")
    u0 = initialize_fem_function(V, "u0")
    u = initialize_fem_function(V, "u")
    f = initialize_fem_function(V, "f")

    if initial_condition is not None:
        u1.interpolate(lambda x: initial_condition.shape(x))

    bc = set_bcs(bcs, domain, V)

    xdmf = initialize_xdmf(domain=domain, output_path=output_path)

    dt = get_resolution(time)

    a, l = wave_equation(
        V=V,
        domain=domain,
        u0=u0,
        u1=u1,
        dt=dt,
        force=f,
        c=c,
        damping_coefficient=damping,
    )
    bilinear_form, linear_form = convert_to_form(input_forms=[a, l])

    if mic_coordinates is not None:
        microphone = MicrophonePressure(domain, np.array(mic_coordinates))
    else:
        microphone = None

    xdmf, p_mic, membrane = time_loop_wave(
        domain=domain,
        bilinear_form=bilinear_form,
        linear_form=linear_form,
        time=time,
        force=force,
        force_shape=shape_profile,
        u=u,
        u0=u0,
        u1=u1,
        force_function=f,
        bc=bc,
        xdmf=xdmf,
        microphone=microphone,
    )
    if microphone:
        save_numpy_file(output_path=output_path.with_suffix(".npy"), data=p_mic)

    if save_membrane:
        r, phi, _, _ = compute_polar_coordinates(V=V)

        r_broadcast = np.broadcast_to(r.reshape(1, -1), membrane.shape)
        phi_broadcast = np.broadcast_to(phi.reshape(1, -1), membrane.shape)
        membrane = np.stack([membrane, r_broadcast, phi_broadcast], axis=-1)

        save_numpy_file(
            output_path=Path(
                output_path.parent, output_path.stem + "r_phi_membrane_whole.npy"
            ),
            data=membrane,
        )
    close_xdmf(xdmf)


def main(
    time_input_path: Path,
    signal_input_path: Path | None,
    initial_condition_path: Path | None,
    loudspeaker_params_path: Path | None,
    output_path: Path,
    mic_coordinates: bool,
    save_membrane: bool,
    mesh_dim: int = 2,
    mesh_resolution: int = 100,
    shape_profile: str | None = None,
) -> None:
    assert isinstance(time_input_path, Path)
    assert isinstance(signal_input_path, Path) or signal_input_path is None
    assert isinstance(initial_condition_path, Path) or initial_condition_path is None
    assert isinstance(loudspeaker_params_path, Path)
    assert isinstance(output_path, Path)
    assert isinstance(mic_coordinates, bool)
    assert isinstance(save_membrane, bool)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    assert isinstance(mesh_dim, int)
    assert isinstance(shape_profile, str) or shape_profile is None

    time = load_numpy_file(time_input_path)
    assert len(time.shape) == 1

    if signal_input_path:
        force = load_numpy_file(signal_input_path)
        assert len(force.shape) == 1
    else:
        force = None

    if initial_condition_path:
        initial_condition = load_numpy_file(initial_condition_path)
        assert len(initial_condition.shape) == 1
    else:
        initial_condition = None

    if force is not None:
        if len(force) > len(time):
            # force = force[:len(time)]
            force = force[len(force) - len(time) :]
        else:
            force = pad_vector(force, time)
    else:
        force = pad_vector(np.array([0]), time)

    assert len(time) == len(force)

    r_diaphragm, r_coil, rho, youngs_modulus, rms, _ = load_loudspeaker_parameters(
        loudspeaker_params_path
    )
    c = np.sqrt(youngs_modulus / rho)  # estimate sound speed inside membrane material
    mesh_resolution = r_diaphragm / 128
    if mesh_dim == 2:
        if shape_profile == "magnetostatic":
            domain = create_circular_mesh(
                radius=r_diaphragm,
                mesh_size=mesh_resolution,
                output_path=output_path,
            )
            if force is not None:
                shape_profile = Shape(
                    name="circle_region",
                    kwargs={"r": r_coil, "center": (0.0, 0.0)},
                    verbose=False,
                )
            if initial_condition is not None:
                initial_condition = Shape(
                    name="radial_spline_profile",
                    kwargs={
                        "r_max": r_diaphragm,
                        "values": initial_condition,
                        "center": (0.0, 0.0),
                    },
                )
        if shape_profile == "dynamic":
            domain = create_annulus_mesh(
                outer_radius=r_diaphragm,
                inner_radius=r_coil,
                mesh_size=mesh_resolution,
                output_path=output_path,
            )
            if force is not None:
                shape_profile = Shape(
                    name="ring_spatial_profile",
                    kwargs={"r": r_coil, "width": 0.01, "center": (0.0, 0.0)},
                    verbose=False,
                )
            if initial_condition is not None:
                initial_condition = Shape(
                    name="radial_spline_profile",
                    kwargs={
                        "r_max": r_diaphragm,
                        "values": initial_condition,
                        "center": (0.0, 0.0),
                    },
                )
    elif mesh_dim == 3:
        raise NotImplementedError
    else:
        raise ValueError

    if mic_coordinates:
        start = np.array([r_coil, 0, 0])
        end = np.array([r_diaphragm, 0, 0])
        num_points = 128
        mic_coordinates = np.linspace(start, end, num_points).T
    else:
        mic_coordinates = None

    # Simulation
    simulate_wave(
        domain=domain,
        time=time,
        force=force,
        initial_condition=initial_condition,
        shape_profile=shape_profile,
        output_path=output_path,
        mic_coordinates=mic_coordinates,
        save_membrane=save_membrane,
        c=c,
        damping=rms,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate membrane motion based on input signal and loudspeaker parameters."
    )
    parser.add_argument(
        "--time_input_path",
        type=str,
        default="examples/time_vector_100ms_48kHz.npy",
        help="Path to the NumPy file containing the time vector.",
    )
    parser.add_argument(
        "--signal_input_path",
        type=str,
        help="Path to the NumPy file containing the input signal. Assigned in the centre of geometry.",
    )
    parser.add_argument(
        "--initial_condition_path",
        type=str,
        help="Path to the NumPy file containing the initial condition. Points from array are scaled and interpolated to geometry.",
    )
    parser.add_argument(
        "--loudspeaker_params_path",
        type=str,
        default="examples/prv_audio_6MB400_8ohm.json",
        help="Path to the JSON file with loudspeaker parameters.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/membrane_motion.xdmf",
        help="Path to the output XDMF file.",
    )
    parser.add_argument(
        "--shape_profile",
        type=str,
        default="magnetostatic",
        choices=["dynamic", "magnetostatic"],
        help="Membrane shape profile (default: dynamic).",
    )
    parser.add_argument(
        "--mesh_dim",
        type=int,
        default=2,
        choices=[2, 3],
        help="mesh dimensionality (default: 2).",
    )
    parser.add_argument(
        "--listen",
        action="store_true",
        help="Listen along membrane radius using MicrophonePressure class.",
    )
    parser.add_argument(
        "--save_membrane",
        action="store_true",
        help="Save membrane snapshots with coords using numpy.",
    )

    args = parser.parse_args()

    if args.signal_input_path is None and args.initial_condition_path is None:
        parser.error(
            "You must provide at least one of --signal_input_path or --initial_condition_path."
        )

    signal_input_path = Path(args.signal_input_path) if args.signal_input_path else None
    initial_condition_path = (
        Path(args.initial_condition_path) if args.initial_condition_path else None
    )

    main(
        time_input_path=Path(args.time_input_path),
        signal_input_path=signal_input_path,
        initial_condition_path=initial_condition_path,
        loudspeaker_params_path=Path(args.loudspeaker_params_path),
        output_path=Path(args.output_path),
        mic_coordinates=args.listen,
        save_membrane=args.save_membrane,
        mesh_dim=args.mesh_dim,
        shape_profile=args.shape_profile,
    )
