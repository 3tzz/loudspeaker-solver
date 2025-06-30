#!/bin/bash

# Help function
usage() {
  echo ""
  echo "Usage: $0 [-h] -s <signal_path.wav> -l <loudspeaker_config_path.json> -c <service_config_path.json>"
  echo ""
  echo "Options:"
  echo "  -s <signal>   Path to input acoustic signal"
  echo "  -l <loudspeaker_config>   Path to loudspeaker config file (required)"
  echo "  -o <output_path>   Output path (required)"
  echo "  -c <service_config>   Path to service config file (optional)"
  echo "  -i <impedance_path>   Path to frequency dependent config file (optional)"
  echo "  -h            Show this help message"
  echo ""
  echo "Positional Arguments:"
  echo "  signal_path.py    Path to signal script"
  echo ""
  exit 1
}


# Optional default value (can be empty or a path)
SERVICE_CONFIG_PATH=""

# Parse options (colon after c means it takes an argument)
while getopts ":s:l:c:o:i:h" opt; do
  case ${opt} in
    s) SIGNAL_PATH="$OPTARG" ;;
    l) LOUDSPEAKER_CONFIG_PATH="$OPTARG" ;;
    o) OUTPUT_PATH="$OPTARG" ;;
    c) SERVICE_CONFIG_PATH="$OPTARG" ;;
    i) IMPEDANCE_PATH="$OPTARG" ;;
    h) usage ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
  esac
done

# Shift out parsed options
shift $((OPTIND - 1))

# Check signal is provided
if [[ -z "$SIGNAL_PATH" ]]; then
  echo "Error: -s <signal_path> is required."
  usage
fi

# Check config is provided
if [[ -z "$LOUDSPEAKER_CONFIG_PATH" ]]; then
  echo "Error: -l <loudspeaker_config_path> is required."
  usage
fi

# Check output path is provided
if [[ -z "$OUTPUT_PATH" ]]; then
  echo "Error: -o <output_path> is required."
  usage
fi

# Debug output
echo "Initializaiton .."
echo "Signal path: $SIGNAL_PATH"
echo "Loudspeaker config path: $LOUDSPEAKER_CONFIG_PATH"
echo "Output path: $OUTPUT_PATH"
if [[ -n "$SERVICE_CONFIG_PATH" ]]; then
  echo "Service config path: $SERVICE_CONFIG_PATH"
else
  echo "Service config: not provided. Running default setup!"
fi

if [[ -n "$" ]]; then
  echo "Impedance path: $IMPEDANCE_PATH"
else
  echo "Impedance path: not provided"
fi

# Create output directory if it doesn't exist
if [[ ! -d "$OUTPUT_PATH" ]]; then
  echo "Creating output directory: $OUTPUT_PATH"
  mkdir -p "$OUTPUT_PATH" || { echo "Failed to create output directory."; exit 1; }
fi

# Create working dir
WORK_DIR="${OUTPUT_PATH}/work_dir"
if [[ ! -d "$WORK_DIR" ]]; then
  mkdir -p "$WORK_DIR" || { echo "Failed to create working directory."; exit 1; }
fi

# Run
echo
echo
echo "Run loudspeaker simulation"
echo "Running Electromagnetic Converter..."
echo "Running Coil Current Converter..."
echo

COIL_PATH="${WORK_DIR}/coil_current.npy"
MODE="frequency_impedance"
MODE="frequency_impedance"

python boomspeaver/electromagnetic/calulate_coil_current.py \
  --input_signal_path $SIGNAL_PATH \
  --loudspeaker_params $LOUDSPEAKER_CONFIG_PATH \
  --output_path $COIL_PATH \
  $MODE \
  --impedance_params $IMPEDANCE_PATH

echo "Running Magnetic Force Converter..."
echo

MAGNETIC_FORCE_PATH="${WORK_DIR}/magnetic_force.npy"

python boomspeaver/electromagnetic/magnetic_force.py \
  --input_signal_path $COIL_PATH \
  --loudspeaker_params $LOUDSPEAKER_CONFIG_PATH \
  --output_path $MAGNETIC_FORCE_PATH

echo "Running Mechanical Converter..."
echo

MECHANICAL_PATH="${WORK_DIR}/mechanical.npy"
MODE="euler"

python boomspeaver/mechanical/oscillation_$MODE.py \
  --input_signal_path $MAGNETIC_FORCE_PATH \
  --loudspeaker_params $LOUDSPEAKER_CONFIG_PATH \
  --output_path $MECHANICAL_PATH

echo "Running Acoustic Converter..."
echo

ACOUSTIC_TEMP="${WORK_DIR}/acoustic/"
if [[ ! -d "$ACOUSTIC_TEMP" ]]; then
  mkdir -p "$ACOUSTIC_TEMP" || { echo "Failed to create working directory."; exit 1; }
fi
ACOUSTIC_PATH="${ACOUSTIC_TEMP}/membrane.xdmf"
MECHANICAL_DISPLACEMENT_PATH="${WORK_DIR}/mechanical.x.npy"
MODE="dynamic"

./run_fenics.sh -c boomspeaver/acoustic/the_membrane.py \
  --time_input_path examples/time_vector_1s_48kHz.npy \
  --signal_input_path $MECHANICAL_DISPLACEMENT_PATH \
  --loudspeaker_params_path $LOUDSPEAKER_CONFIG_PATH \
  --output_path $ACOUSTIC_PATH \
  --shape_profile $MODE \
  --listen