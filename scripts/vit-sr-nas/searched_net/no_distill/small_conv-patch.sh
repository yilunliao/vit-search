#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'main.py' | awk '{print $2}')

IMAGENET_PATH="/code/imagenet"

python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --model flexible_vit_sr_patch14_224_patch_output \
    --batch-size 128 \
    --data-path $IMAGENET_PATH \
    --epochs 300 \
    --num_workers 8 \
    --output_dir models/vit-sr-nas/searched_net/small_conv-patch/example_per_arch@64/mac@2.9G \
    --val-bs 192 \
    --network-def \
    --no-repeated-aug \
    --use-patch-mixup \
    --drop-path 0.3