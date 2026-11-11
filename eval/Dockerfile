# Evaluation image for the released SAIR checkpoints. Build from the repository root:
#   docker build -f eval/Dockerfile -t sair-eval .
#   docker run --gpus all -it sair-eval
# Python 3.8.10 as in the paper; DAVIS 2017 is downloaded to /data/DAVIS for the video backgrounds.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget unzip git build-essential libffi-dev libssl-dev zlib1g-dev libbz2-dev liblzma-dev \
        libsqlite3-dev libreadline-dev tk-dev uuid-dev libgl1 libegl1 libglfw3 libosmesa6 patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz \
    && tar xzf Python-3.8.10.tgz && cd Python-3.8.10 \
    && ./configure --prefix=/opt/python3.8 && make -j"$(nproc)" && make install \
    && cd .. && rm -rf Python-3.8.10 Python-3.8.10.tgz
ENV PATH=/opt/python3.8/bin:$PATH
RUN ln -s /opt/python3.8/bin/python3.8 /usr/local/bin/python && python -m pip install --upgrade pip

RUN wget -q https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -O /tmp/davis.zip \
    && mkdir -p /data && unzip -q /tmp/davis.zip -d /data && rm /tmp/davis.zip

WORKDIR /app
COPY . /app
RUN pip install -e . && pip install -r eval/requirements.txt
WORKDIR /app/eval
