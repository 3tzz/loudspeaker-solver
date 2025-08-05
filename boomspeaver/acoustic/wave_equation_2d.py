# pylint: disable=invalid-name
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-arguments
# pylint: disable=redefined-outer-name

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
    get_resolution,
    load_numpy_file,
    pad_vector,
    save_numpy_file,
)
from boomspeaver.tools.fem import (
    Shape,
    close_xdmf,
    convert_to_form,
    initialize_fem_function,
    initialize_xdmf,
    load_mesh,
    set_bcs,
    wave_equation,
)


def time_loop_wave(
    domain: mesh.Mesh,
    bilinear_form: ufl.form.Form,
    linear_form: ufl.form.Form,
    time: np.ndarray,
    force: np.ndarray,
    force_shape: Shape | None,
    u: fem.function.Function,
    u0: fem.function.Function,
    u1: fem.function.Function,
    force_function: fem.function.Function,
    bc: Any,
    xdmf: io.utils.XDMFFile,
    microphone: MicrophonePressure | None,
) -> tuple[io.utils.XDMFFile, np.ndarray]:
    """Time step loop simulation."""
    force = pad_vector(force, time)
    p_mic = np.zeros_like(time)
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
    force_shape: Shape | None,
    mic_coordinates: list[float] | None,
    bcs: str = "neumann",
):
    """Setup and Simulate wave equation."""

    V = fem.functionspace(domain, ("Lagrange", 1))

    u1 = initialize_fem_function(V, "u1")
    u0 = initialize_fem_function(V, "u0")
    u = initialize_fem_function(V, "u")
    f = initialize_fem_function(V, "f")

    bc = set_bcs(bcs, domain, V)

    xdmf = initialize_xdmf(domain=domain, output_path=output_path)

    dt = get_resolution(time)
    a, l = wave_equation(V=V, domain=domain, u0=u0, u1=u1, dt=dt, force=f)

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
        force_shape=force_shape,
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
    mesh_path: Path | None,
    time_path: Path,
    force_path: Path | None,
    output_path: Path,
    mic_coordinates: list[float] | None,
    force_shape: Shape | None,
    bcs: str = "neumann",
):
    """Load input data and simulate."""
    assert isinstance(mesh_path, Path) or mesh_path is None
    assert isinstance(time_path, Path)
    assert isinstance(force_path, Path) or force_path is None
    assert isinstance(output_path, Path)
    assert isinstance(mic_coordinates, list) or mic_coordinates is None
    assert isinstance(force_shape, Shape) or force_shape is None
    assert isinstance(bcs, str)

    domain = load_mesh(mesh_path)
    time = load_numpy_file(time_path)

    if force_path is not None:
        force = load_numpy_file(force_path)
    else:
        force = np.array([1.0])

    simulate_wave(
        domain=domain,
        time=time,
        force=force,
        output_path=output_path,
        bcs=bcs,
        mic_coordinates=mic_coordinates,
        force_shape=force_shape,
    )


if __name__ == "__main__":
    repo_dir = get_repo_dir()

    # Example
    time_path = repo_dir / "examples/time_vector.npy"
    # time_path = repo_dir / "examples/time_vector_1s_48kHz.npy"
    mesh_path = None

    output_path = repo_dir / "output/wave_equation_default.xdmf"
    # output_path = repo_dir / "output/wave_equation_default_440hz_highresolution.xdmf"
    # output_path = repo_dir / "output/wave_equation_default_neumann.xdmf"

    force_shape = Shape(
        name="square_spatial_profile",
        kwargs={"center": (-10.0, 0.0), "shape": (0.0000001, 0.5)},
        verbose=False,
    )
    mic_coordinates = [0.0, 0.0, 0.0]

    # Default Neumann Example
    bcs = "neumann"

    # Default Dirichlet Example
    bcs = "dirichlet"

    # Default External Force
    force_path = repo_dir / "examples/cosine_wave.npy"
    # force_path = None

    main(
        mesh_path=mesh_path,
        time_path=time_path,
        force_path=force_path,
        output_path=output_path,
        mic_coordinates=mic_coordinates,
        force_shape=force_shape,
        bcs=bcs,
    )
