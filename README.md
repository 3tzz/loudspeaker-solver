# loudspeaker-solver

<p align="left">
  <img src="./dosc/beaver_chatgpt_generated.png" width="400">
</p>

## Environment Setup

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

## Usage

### Run any repo script

#### Dolfinx numeric real mode

```bash
cd &{ROOT_DIR}
./run.sh -c path/to/repository/relative/python_script.py
```

#### Dolfinx numeric complex mode

```bash
cd &{ROOT_DIR}
./run.sh path/to/repository/relative/python_sctipt.py
```

### Loudspeaker solver service

### Repository

#### Loudspeaker

Represents loudspeaker parameters. Good resources u can find in [loudspeaker database site](https://loudspeakerdatabase.com).
Example loudspeaker `PRV Audio 6MB400` file with parameters created in [repository](./examples/prv_audio_6MB400_8ohm.json) from [site](https://loudspeakerdatabase.com/PRV/6MB400).

- `schema.py` - structure representing loudspeaker parameters
- `geometry`
- `mesh`

#### Electromagnetic

#### Mechanical

#### Acoustic

#### Tools
