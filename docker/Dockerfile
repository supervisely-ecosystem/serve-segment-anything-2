FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip wget curl -y
RUN apt-get install -y git

RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install packaging==22.0

ENV TORCH_CUDA_ARCH_LIST="8.6 8.9"
ENV CUDA_HOME="/usr/local/cuda-12.1"

RUN git clone https://github.com/facebookresearch/sam2.git
WORKDIR /sam2
RUN SAM2_BUILD_ALLOW_ERRORS=0 pip3 install -v -e ".[demo]"
RUN python3 setup.py clean --all
RUN python3 setup.py build_ext --inplace

RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN pip3 install opencv-python
RUN pip3 install mock==5.1.0

RUN mkdir -p /sam2.1_weights

# Download SAM 2.1 checkpoints
RUN curl -L -o /sam2.1_weights/sam2.1_hiera_tiny.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
RUN curl -L -o /sam2.1_weights/sam2.1_hiera_small.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
RUN curl -L -o /sam2.1_weights/sam2.1_hiera_base_plus.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
RUN curl -L -o /sam2.1_weights/sam2.1_hiera_large.pt "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"

RUN pip3 install git+https://github.com/supervisely/supervisely.git@tracking-no-api
# RUN pip3 install supervisely==6.73.233
LABEL python_sdk_version=6.73.275
