FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git

RUN pip3 install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install packaging==22.0

ENV TORCH_CUDA_ARCH_LIST "8.9"
ENV CUDA_HOME /usr/local/cuda-12.1

RUN git clone https://github.com/facebookresearch/segment-anything-2.git
WORKDIR /segment-anything-2
RUN git reset --hard d421e0b
RUN SAM2_BUILD_ALLOW_ERRORS=0 pip3 install -v -e ".[demo]"
RUN python3 setup.py clean --all
RUN python3 setup.py build_ext --inplace

RUN python3 -m pip install git+https://github.com/supervisely/supervisely.git@cache-error-fix-3

RUN apt-get install ffmpeg libgeos-dev libsm6 libxext6 libexiv2-dev libxrender-dev libboost-all-dev -y
RUN pip install opencv-python

RUN pip3 install mock==5.1.0

RUN mkdir -p sam2_weights
ADD sam2_weights /sam2_weights/

LABEL python_sdk_version=6.73.165