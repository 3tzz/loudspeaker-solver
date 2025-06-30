#!/bin/bash

# Default value for complex flag (1 = on, 0 = off)
COMPLEX_MODE=1

# Parse arguments
while getopts ":c" opt; do
  case ${opt} in
  c)
    COMPLEX_MODE=0
    ;;
  \?)
    echo "Usage: $0 [-c to disable complex mode] <path_to_script.py>"
    exit 1
    ;;
  esac
done

# Validate
shift $((OPTIND - 1))
if [ -z "$1" ]; then
  echo "Usage: $0 [-c to disable complex mode] <path_to_script.py>"
  exit 1
fi

# Run
if [ $COMPLEX_MODE -eq 1 ]; then
  echo "Complex mode"
  docker compose exec fenics bash -c "source /usr/local/bin/dolfinx-complex-mode && python3 $1"
else
  echo "Real mode"
  docker compose exec fenics bash -c "source /usr/local/bin/dolfinx-real-mode && python3 $1"
fi
