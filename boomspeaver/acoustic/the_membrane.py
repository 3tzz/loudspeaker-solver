# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals

from pathlib import Path

import numpy as np
import argparse
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem

from mpi4py import MPI
from petsc4py import PETSc

from boomspeaver.acoustic.microphone_pressure import MicrophonePressure
from boomspeaver.loudspeaker.schema import Loudspeaker
from boomspeaver.tools.data import get_repo_dir, load_numpy_file, pad_vector, save_numpy_file
from boomspeaver.tools.fem import (
    Shape,
    close_xdmf,
    convert_to_form,
    create_circular_mesh,
    create_annulus_mesh,
    initialize_fem_function,
    initialize_xdmf,
    set_bcs,
    wave_equation,
)
from boomspeaver.tools.signal.signal import get_resolution


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
    return r_diaphragm, r_coil, rho, youngs_modulus, rms


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
) -> tuple[io.utils.XDMFFile, np.ndarray]:
    """Time step loop simulation."""
    force = pad_vector(force, time)
    p_mic = np.empty(len(time), dtype=object)

    for idx, (t, fv) in enumerate(zip(time, force)):
        print(idx, t, fv)
        if force_shape is None:
            force_function.interpolate(lambda x: np.full(x.shape[1], fv))
        else:
            assert isinstance(force_shape, Shape)
            force_function.interpolate(lambda x: fv * force_shape.shape(x))

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
    time: np.ndarray,
    force: np.ndarray,
    output_path: Path,
    shape_profile: Shape | None,
    c: float,
    mic_coordinates: list[float] | None,
    damping: float | None = None,
    bcs: str = "dirichlet",
) -> None:
    """Setup and simulate wave equation."""
    V = fem.functionspace(domain, ("Lagrange", 1))

    u1 = initialize_fem_function(V, "u1")
    u0 = initialize_fem_function(V, "u0")
    u = initialize_fem_function(V, "u")
    f = initialize_fem_function(V, "f")

    bcs = None

    bc = set_bcs(bcs, domain, V)

    xdmf = initialize_xdmf(domain=domain, output_path=output_path)

    dt = get_resolution(time)
    print(dt)

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

    xdmf, p_mic = time_loop_wave(
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
    save_numpy_file(output_path=output_path.with_suffix(".npy"), data=p_mic)
    close_xdmf(xdmf)


def main(
    time_input_path: Path,
    signal_input_path: Path,
    loudspeaker_params_path: Path | None,
    output_path: Path,
    mic_coordinates: bool,
    mesh_dim: int = 2,
    shape_profile: str | None = None,
) -> None:
    assert isinstance(time_input_path, Path)
    assert isinstance(signal_input_path, Path)
    assert isinstance(loudspeaker_params_path, Path)
    assert isinstance(output_path, Path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    assert isinstance(mesh_dim, int)
    assert isinstance(shape_profile, str) or shape_profile is None

    time = load_numpy_file(time_input_path)
    force = load_numpy_file(signal_input_path)
    force = pad_vector(force, time)
    time = time[:1000]  # TODO: just for prototyping
    force = force[:1000]  # TODO: just for prototyping
    print(len(time))
    print(len(force))

    assert len(time) == len(force)

    r_diaphragm, r_coil, rho, youngs_modulus, rms = load_loudspeaker_parameters(
        loudspeaker_params_path
    )
    c = np.sqrt(youngs_modulus / rho)  # estimate sound speed inside membrane material

    if mesh_dim == 2:
        if shape_profile == "magnetostatic":
            domain = create_circular_mesh(
                radius=r_diaphragm,
                mesh_size=r_diaphragm / 100,
                output_dir=output_path.parent,
            )
        if shape_profile == "dynamic":
            domain = create_annulus_mesh(
                outer_radius=r_diaphragm,
                inner_radius=r_coil,
                mesh_size=r_diaphragm / 100,
                output_dir=output_path.parent,
            )
    elif mesh_dim == 3:
        raise NotImplementedError
    else:
        raise ValueError

    shape_profile = Shape(
        name="ring_spatial_profile",
        kwargs={"r": r_coil, "width": 0.01, "center": (0.0, 0.0)},
        verbose=False,
    )

    if mic_coordinates:
        # mic_coordinates = [r_diaphragm-r_coil, 0.0, 0.0]

        start = np.array([r_coil, 0, 0])
        end = np.array([r_diaphragm, 0, 0])
        num_points = 100
        mic_coordinates = np.linspace(start, end, num_points).T
    else:
        mic_coordinates= None

    # Simulation
    simulate_wave(
        domain=domain,
        time=time,
        force=force,
        shape_profile=shape_profile,
        output_path=output_path,
        mic_coordinates=mic_coordinates,
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
        default="examples/time_vector_1s_48kHz.npy",
        help="Path to the NumPy file containing the time vector.",
    )
    parser.add_argument(
        "--signal_input_path",
        type=str,
        default="examples/cosine_wave.npy",
        help="Path to the NumPy file containing the input signal.",
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
        default="dynamic",
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

    args = parser.parse_args()

    main(
        time_input_path=Path(args.time_input_path),
        signal_input_path=Path(args.signal_input_path),
        loudspeaker_params_path=Path(args.loudspeaker_params_path),
        output_path=Path(args.output_path),
        mic_coordinates=args.listen,
        mesh_dim=args.mesh_dim,
        shape_profile=args.shape_profile,
    )