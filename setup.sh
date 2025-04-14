#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda --version

# Create environment
#conda env remove --name fenics
conda create -n fenics python=3.11
conda activate fenics

# Jupyter setup
conda install -c conda-forge jupyterlab
conda install -c conda-forge catppuccin-jupyterlab

# Install fenics
conda install -c conda-forge fenics-dolfinx mpich pyvista
conda install cudatoolkit=11.8 cuda-version=11

# Install gmsh (mesh genertor)
# pip  install gmsh
