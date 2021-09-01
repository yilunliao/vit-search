#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'evo_search.py' | awk '{print $2}')

IMAGENET_PATH="/code/imagenet"
MODEL_PATH="models/vit-sr-nas/super_net/small_conv-patch/example_per_arch@64/epoch@119_checkpoint.pth.tar"

python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env evo_search.py \
    --data-path $IMAGENET_PATH \
    --val-bs 256 \
    --num_workers 8 \
    --model-path $MODEL_PATH \
    --model 'flexible_vit_sr_patch14_224_patch_output' \
    --network-def '((4, 320), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (3, 320, 640), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (3, 640, 1280), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (2, 1280, 1000))' \
    --search-space 'sr_small_mh' \
    --constraint-value 2900000000 \
    --search-iter 20 \
    --parent-size 75 \
    --init-popu-size 500 \
    --mutate-size 75 \
    --output_dir models/vit-sr-nas/evolutionary_search/small_conv-patch/example_per_arch@64/mac@2.9G
    
  