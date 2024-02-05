# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

RUN apt-get update && apt-get install -y software-properties-common git nvidia-cuda-toolkit python3-pip nano

WORKDIR /workspace

RUN pip3 install --upgrade pip  && \
    pip install opencv-python==4.8.0.76 && \
    pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# Clone the repos
RUN git clone https://github.com/facebookresearch/detectron2.git 

# Install detectron2
RUN python3 -m pip install -e detectron2

# Fix various errors regarding version mismatch
RUN pip3 install seaborn && \
    pip3 uninstall -y matplotlib && \
    pip3 uninstall -y shapely && \
    pip3 install matplotlib==3.7.3 && \
    pip3 install shapely==1.8.0 

# Install missing packages required for training this specific model    
RUN pip3 install pip install omegaconf 
RUN pip3 install timm
    
RUN apt-get autoremove -y && \
    apt-get autoclean -y && \
    rm -rf /var/lib/apt/lists/*
