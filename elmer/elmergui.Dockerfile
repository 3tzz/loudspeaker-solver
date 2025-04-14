
FROM ubuntu:focal

# Set the working directory to /home
WORKDIR /home

# Create an apt configuration file to fix erroneous "hash sum mismatch" errors
RUN printf "Acquire::http::Pipeline-Depth 0;\nAcquire::http::No-Cache true;\nAcquire::BrokenProxy true;" \
	>> /etc/apt/apt.conf.d/99fixbadproxy

ENV DEBIAN_FRONTEND="noninteractive"

# Packages for Elmer/Ice
RUN apt update && apt upgrade -y

RUN apt install -y --no-install-recommends \
  build-essential \
  git \
  libblas-dev \
  liblapack-dev \
  libmumps-dev \
  libparmetis-dev \
  libhypre-dev \
  gfortran \
  mpich \
  sudo \
  cmake \
  ca-certificates \
  less

# Packages for ElmerGUI
RUN apt-get install -y liboce-modeling-dev \
  liboce-foundation-dev \
  qtscript5-dev \
  libqt5script5 \
  libqt5widgets5 \
  libqt5core5a \
  libqt5gui5 \
  libqt5help5 \
  libqt5opengl5 \
  libqt5opengl5-dev \
  libqt5svg5-dev \
  libvtk6.3 \
  libvtk6-dev \
  libvtk6.3-qt \
  libvtk6-qt-dev \
  libqwt-qt5-dev

# Clone the ElmerIce source code and compile Elmer/Ice
RUN git clone https://www.github.com/ElmerCSC/elmerfem elmer \
        && mkdir elmer/builddir \
	&& cd elmer/builddir \
	&& cmake /home/elmer \
		-DCMAKE_INSTALL_PREFIX=/usr/local/Elmer-devel \
		-DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DWITH_OpenMP:BOOLEAN=TRUE \
		-DCMAKE_Fortran_COMPILER=/usr/bin/gfortran \
		-DWITH_MPI:BOOL=TRUE -DWITH_Mumps:BOOL=TRUE \
		-DWITH_Hypre:BOOL=FALSE -DWITH_Trilinos:BOOL=FALSE \
		-DWITH_ELMERGUI:BOOL=TRUE -DWITH_ElmerIce:BOOL=TRUE \
                -DWITH_LUA:BOOL=TRUE \
	&& make \
	&& make install

# Set the path
ENV PATH=$PATH:/usr/local/Elmer-devel/bin
ENV PATH=$PATH:/usr/local/Elmer-devel/share/elmersolver/lib

# Add user
ENV USER=freetzz
RUN adduser --disabled-password --gecos '' ${USER} \
	&& adduser ${USER} sudo \
	&& echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER ${USER}
ENV HOME=/home/${USER}
WORKDIR ${HOME}
