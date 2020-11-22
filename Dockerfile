FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN apt-get update || true && apt-get install -y build-essential \
    wget curl git git-lfs vim zip unzip tmux htop software-properties-common \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
    libssl-dev libsqlite3-dev libreadline-dev libffi-dev libbz2-dev \
    python3.7 python3-pip python3.7-venv python3.7-dev

RUN add-apt-repository ppa:deadsnakes/ppa -y \
    && apt install python3.7 -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python get-pip.py \
    && rm get-pip.py

WORKDIR /notebook
COPY pyproject.toml poetry.lock /notebook/

RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev

# RUN git clone https://github.com/FilatovArtm/DeslantImg.git /home/DeslantImg && cd /home/DeslantImg && ./build.sh && cd ..

RUN apt update -y && apt install libgl1-mesa-glx -y
RUN pip install symspellpy