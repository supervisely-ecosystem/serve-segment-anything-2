FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    curl \
    git \
    ffmpeg \
    libgeos-dev \
    libsm6 \
    libexiv2-dev \
    libxrender-dev \
    libboost-all-dev \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir packaging==22.0

ENV TORCH_CUDA_ARCH_LIST="8.6 8.9"
ENV CUDA_HOME="/usr/local/cuda-12.1"

RUN git clone https://github.com/facebookresearch/sam2.git
WORKDIR /sam2
RUN SAM2_BUILD_ALLOW_ERRORS=0 pip3 install --no-cache-dir -v -e ".[demo]" && rm -rf /root/.cache/pip
RUN python3 setup.py clean --all
RUN python3 setup.py build_ext --inplace && rm -rf /root/.cache/pip

RUN pip3 install --no-cache-dir opencv-python
RUN pip3 install --no-cache-dir mock==4.0.2
RUN pip3 install --no-cache-dir segmentation-models-pytorch==0.3.4

RUN mkdir -p /sam2.1_weights

# Download SAM 2.1 checkpoints
RUN curl -L -o /sam2.1_weights/sam2.1_hiera_tiny.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
RUN curl -L -o /sam2.1_weights/sam2.1_hiera_small.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
RUN curl -L -o /sam2.1_weights/sam2.1_hiera_base_plus.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
RUN curl -L -o /sam2.1_weights/sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

RUN python3 -m pip install --no-cache-dir supervisely==6.73.479

LABEL python_sdk_version=6.73.479
