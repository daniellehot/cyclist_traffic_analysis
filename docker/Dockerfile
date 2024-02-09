# Use an official NVIDIA CUDA base image
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

WORKDIR /repos

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common 
RUN apt-get install -y git 
RUN apt-get install -y python3-pip 
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install ninja
RUN git clone https://github.com/IDEA-Research/DINO.git
RUN pip3 install -r DINO/requirements.txt
COPY compile_MultiScaleDeformableAttention.sh /repos/
RUN pip3 install opencv-python
RUN apt-get -y autoremove 
RUN apt-get -y autoclean 
