from setuptools import find_packages, setup

setup(
    name="boomspeaver",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib",
        "pyvistaqt",
        "pyvista",
        "imageio",
        "gmsh",
        "soundfile",
        "scipy",
        "dotenv",
        "librosa",
        "pandas",
    ],
)
