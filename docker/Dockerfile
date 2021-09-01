FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# Set up the volume permission
RUN set -eux; \
    apt-get update; \
    apt-get install -y gosu=1.10-1; \
    rm -rf /var/lib/apt/lists/*;
COPY init_volume.sh /usr/local/bin/init_volume.sh
RUN chmod +x /usr/local/bin/init_volume.sh
ENTRYPOINT ["/usr/local/bin/init_volume.sh"]

# Install basic tools
RUN apt-get update; \
    apt-get install -y vim wget 

# Install the dependencies of training
RUN pip install tqdm==4.45.0 tensorboardX==2.0 scipy==1.5.1 matplotlib==3.3.0 scikit-learn==0.23.1 
#RUN pip install pyqt5==5.11.2
#RUN pip install --upgrade PyQtWebEngine
#RUN pip install spyder
#RUN apt-get install python-pyqt5.qtwebkit

RUN pip install torch==1.7.0 torchvision==0.8.1 timm==0.3.2

#RUN pip install https://developer.download.nvidia.com/compute/redist/cuda/10.0/nvidia-dali/nvidia_dali-0.20.0-1177723-cp37-cp37m-manylinux1_x86_64.whl
#RUN git clone https://github.com/NVIDIA/apex /tmp/apex && \
#    cd /tmp/apex && \
#    git checkout ca00adace51ae4d82d2ff5cfc1ef1c9174ff6aaa && \
#    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ && \
#    rm -rf /tmp/apex
