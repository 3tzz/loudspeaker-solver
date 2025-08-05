ARG ROOT_DIR
ARG IMAGE_NAME
FROM ${IMAGE_NAME}

RUN apt update && \
    apt install -y libgl1 libxkbcommon-x11-0 xvfb

WORKDIR ${ROOT_DIR}

COPY . ${ROOT_DIR}

RUN source /usr/local/bin/dolfinx-complex-mode && pip install -e .
RUN source /usr/local/bin/dolfinx-real-mode && pip install -e .
