FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt update

RUN apt-get install -y python3 python3-pip ffmpeg libsm6 libxext6 git wget curl

RUN pip3 install poetry

WORKDIR /app

# Poetry dependencies
COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

# Grounding DINO
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX;8.9;9.0"
WORKDIR /app/packages
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git
WORKDIR /app/packages/GroundingDINO
RUN python3 setup.py install

# Weights
WORKDIR /app/weights
RUN mkdir -p /app/weights/dino
# Dino weights
WORKDIR /app/weights/dino
RUN cp ../../packages/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py .
RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
# SAM weights
WORKDIR /app/weights
RUN mkdir -p /app/weights/sam
WORKDIR /app/weights/sam
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

WORKDIR /app
EXPOSE 8000
COPY . /app
ENV CUDA_HOME=/usr/local/cuda
CMD ["uvicorn", "app:app", "--host", "0.0.0.0" ]
