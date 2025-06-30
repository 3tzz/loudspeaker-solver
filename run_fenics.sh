#!/bin/bash

# Default value for complex flag (1 = on, 0 = off)
COMPLEX_MODE=1

# Parse optional flags
while getopts ":c" opt; do
  case ${opt} in
    c)
      COMPLEX_MODE=0
      ;;
    \?)
      echo "Usage: $0 [-c to disable complex mode] <script.py> [args...]"
      exit 1
      ;;
  esac
done

# Shift to remove parsed options (e.g. -c)
shift $((OPTIND - 1))

# Validate: first argument must be the script path
SCRIPT_PATH=$1
if [ -z "$SCRIPT_PATH" ]; then
  echo "Usage: $0 [-c to disable complex mode] <script.py> [args...]"
  exit 1
fi

# Shift again to remove the script path from $@
shift 1

# Build command
if [ $COMPLEX_MODE -eq 1 ]; then
  echo "Complex mode"
  docker compose exec fenics bash -c "source /usr/local/bin/dolfinx-complex-mode && python3 $SCRIPT_PATH $*"
else
  echo "Real mode"
  docker compose exec fenics bash -c "source /usr/local/bin/dolfinx-real-mode && python3 $SCRIPT_PATH $*"
fi
