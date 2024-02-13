# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /workspace

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common git python3-pip ffmpeg libsm6 libxext6 nano
RUN pip3 install ninja
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 uninstall -y pillow && pip3 install pillow==9.5.0
COPY run_dino_demo_detrex.sh /workspace/
COPY install.sh /workspace/

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

#ENV TORCH_CUDA_ARCH_LIST=Turing
RUN git clone https://github.com/IDEA-Research/detrex.git
RUN cd detrex && \
    git submodule init && \
    git submodule update && \
    python3 -m pip install -e detectron2
RUN cd detrex && pip3 install -e .

RUN apt-get -y autoremove 
RUN apt-get -y autoclean 