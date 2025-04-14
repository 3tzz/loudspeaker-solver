#!/bin/bash

# Default value for complex flag (1 = on, 0 = off)
COMPLEX_MODE=1

# Parse command-line arguments
while getopts ":c" opt; do
  case ${opt} in
  c)
    COMPLEX_MODE=0 # Disable complex mode if -c is passed
    ;;
  \?)
    echo "Usage: $0 [-c to disable complex mode] <path_to_script.py>"
    exit 1
    ;;
  esac
done

# Shift the arguments to remove the option flags (like -c)
shift $((OPTIND - 1))

# Check if an argument (path to Python script) is provided
if [ -z "$1" ]; then
  echo "Usage: $0 [-c to disable complex mode] <path_to_script.py>"
  exit 1
fi

docker exec fenics_container bash -c "apt update && apt install -y libgl1 libxkbcommon-x11-0 xvfb && pip install pyvista pyvistaqt imageio"
# Run the Docker command with the provided file and complex flag
if [ $COMPLEX_MODE -eq 1 ]; then
  echo "Complex mode"
  docker compose exec fenics bash -c "source /usr/local/bin/dolfinx-complex-mode && python3 $1"
else
  echo "Real mode"
  docker compose exec fenics bash -c "python3 $1"
fi
