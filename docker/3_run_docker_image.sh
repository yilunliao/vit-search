#!/bin/bash
docker run --gpus all --rm -it --shm-size=256g -e PUID=$(id -u $(logname)) -v /data:/data -v /home:/code deit_vit:v1 /bin/bash
