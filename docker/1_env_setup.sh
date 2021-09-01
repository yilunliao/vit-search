#!/bin/bash

# First, remove old versions of Docker
apt-get remove -y docker docker-engine docker.io

# Next, add Docker Repository
apt-get update
apt-get install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL 'https://download.docker.com/linux/ubuntu/gpg' | sudo apt-key add -
add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu  $(lsb_release -cs) stable"

# Install Docker CE
apt-get update
apt-get install -y docker-ce

# Verify the Docker installation
# sudo docker run hello-world

# Install nvidia-docker2 and restart the Docker daemon 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update && sudo apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Verify the nvidia-docker2 installation
#docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
