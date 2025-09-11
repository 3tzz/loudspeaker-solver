# loudspeaker-solver

<p align="left">
  <img src="./docs/beaver_chatgpt_generated.png" width="400">
</p>

## Environment Setup

```bash
cd ${ROOT_DIR}
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure

Configure variables in [.env file](./.env) if changes are needed.

### Build image & Run container

Image contains [FEM dolfinx](https://github.com/FEniCS/dolfinx) and necessary python packages.

```bash
cd ${ROOT_DIR}
docker compose up -d
```

### Close

Remember to clean exited docker contained using:

```bash
cd ${ROOT_DIR}
docker compose down
```

#### Known Issue

```bash
docker exec -it fenics /bin/bash
pip install -e .
exit
```

## Usage

```bash
cd &{ROOT_DIR}
source .venv/bin/activate
./run.sh -s examples/cosine_wave.wav  -l examples/prv_audio_6MB400_8ohm.json -o output -i examples/electrical_impedance.csv
```

### Run fenics scripts

#### Dolfinx numeric real mode

```bash
cd &{ROOT_DIR}
./run_fenics.sh -c path/relative/to/repository/python_script.py
```

#### Dolfinx numeric complex mode

```bash
cd &{ROOT_DIR}
./run_fenics.sh path/relative/to/repository/python_script.py
```

## Repository Structure

### Loudspeaker

Represents loudspeaker parameters. Real world resources u can find in [loudspeaker database site](https://loudspeakerdatabase.com).
Example loudspeaker `PRV Audio 6MB400` file with parameters created in [repository](./examples/prv_audio_6MB400_8ohm.json) from [site](https://loudspeakerdatabase.com/PRV/6MB400).

- `schema.py` – structure representing loudspeaker parameters.
- `geometry` – building geometry for **FreeCAD** python console API. Export parts manually in `.brep` extension.
  - `create_driver_prv6MB400_only.py`
  - `create_driver_prv6MB400_in_room.py`
- `mesh` – create geometry mesh for **FEniCS** fem scripts
  - `loudspeaker_in_room_2D.py`
- `electrical_impedance.py` – structure representing frequency dependent impedance eg. `examples/electrical_impedance.jpeg`

### Electromagnetic

Represents loudspeaker electromagnetic converter part that transforms **audio signal** to **electric current** at first and then to **magnetic force**. According to loudspeaker parameters from [loudspeaker database site](https://loudspeakerdatabase.com).

- `calculate_coil_current.py` – signal voltage-to-current converter for loudspeaker signals
- `magnetic_force.py` – signal current-to-magnetic force converter for loudspeaker signals

### Mechanical

Represents loudspeaker mechanical converter. This part transforms **magnetic force** to **mechanical oscillation**. According to loudspeaker parameters from [loudspeaker database site](https://loudspeakerdatabase.com).

- `oscillation_euler.py` – signal magnetic_force-to-membrane_oscillation converter for loudspeaker signals. ODE numeric solver using Euler Forward method.
- `oscillation_radau.py` – signal magnetic_force-to-membrane_oscillation converter for loudspeaker signals. ODE numeric solver using Radau implicit method.
- `oscillation_analytical_fourier.py` – signal magnetic_force-to-membrane_oscillation converter for loudspeaker signals. Analytical solver using STFT signal decomposition and analytical formulas.

### Structural

- `fem/the_membrane.py` - signal membrane_oscillation-to-membrane_propagation converter for loudspeaker signals. FEM solver using wave equation.
- `surrogate/` — Code for training surrogate models. Currently serves as infrastructure for training, but may need refinement and will likely be relocated in the future.

### Acoustic

- `microphone_pressure.py` - part of code for grasping data from defined points from simulation domain using cartesian coordinates
- `wave_equation_2d.py` - signal membrane_oscillation-to-room_sound_propagation converter for loudspeaker signal. FEM solver but only 2d prototype.
- `wave_equation_piston_3d.py` - signal membrane_oscillation-to-room_sound_propagation converter for loudspeaker signal. FEM solver using wave equation.
- `wave_equation_mdynamic_3d.py` - signal membrane_propagation-to-room_sound_propagation converter for loudspeaker signal. FEM solver using wave equation from structural results.

### Tools

- `dsp` — utilities and functions for digital signal processing.
- `plot` - custom plotting functions
- `data.py` - utilities for data management, including loading, saving, and locating appropriate.
- `fem.py` - methods used in fem scripts
- `multiprocessing_runner.py` - class for multiprocessing run
- `sampler.py` - script for generating random membrane initial states.
