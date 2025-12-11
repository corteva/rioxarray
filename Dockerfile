ARG GDAL=ubuntu-full-3.8.5
FROM ghcr.io/osgeo/gdal:${GDAL}
ARG PYTHON_VERSION="3.12"
ENV LANG="C.UTF-8"
ENV LC_ALL="C.UTF-8"
ENV PIP_NO_BINARY="rasterio"
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    g++ \
    gdb \
    make \
    python3-pip \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ADD . /app
RUN python${PYTHON_VERSION} -m venv /venv && \
    /venv/bin/python -m pip install -U pip && \
    /venv/bin/python -m pip install -e .[dev,doc,test]
ENTRYPOINT ["/bin/bash", "-c"]
