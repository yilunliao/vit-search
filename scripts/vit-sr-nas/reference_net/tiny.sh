#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'main.py' | awk '{print $2}')

IMAGENET_PATH="/data/datasets/imagenet-fast/data/imagenet"

python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --model flexible_vit_sr_patch14_224_patch_output \
    --batch-size 128 \
    --data-path $IMAGENET_PATH \
    --epochs 300 \
    --num_workers 12 \
    --output_dir models/vit-sr-nas/reference_net/tiny \
    --val-bs 192 \
    --network-def '((4, 192), (1, (192, 3, 64), (192, 768), 1), (1, (192, 3, 64), (192, 768), 1), (1, (192, 3, 64), (192, 768), 1), (1, (192, 3, 64), (192, 768), 1), (3, 192, 384), (1, (384, 6, 64), (384, 1536), 1), (1, (384, 6, 64), (384, 1536), 1), (1, (384, 6, 64), (384, 1536), 1), (1, (384, 6, 64), (384, 1536), 1), (3, 384, 768), (1, (768, 12, 64), (768, 3072), 1), (1, (768, 12, 64), (768, 3072), 1), (1, (768, 12, 64), (768, 3072), 1), (1, (768, 12, 64), (768, 3072), 1), (2, 768, 1000))' \
    --no-repeated-aug \
    --use-patch-mixup \
    --drop-path 0.2
    
