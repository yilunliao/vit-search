#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'evo_search.py' | awk '{print $2}')

IMAGENET_PATH="/data/datasets/imagenet-fast/data/imagenet"
MODEL_PATH="models/vit-sr-nas/super_net/tiny_666/no_distill/single_arch/epoch@119_checkpoint.pth.tar"

python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env evo_search.py \
    --data-path $IMAGENET_PATH \
    --val-bs 256 \
    --num_workers 16 \
    --model-path $MODEL_PATH \
    --model 'flexible_vit_sr_patch14_224' \
    --network-def '((0, 256), (1, (256, 4, 64), (256, 896), 1), (1, (256, 4, 64), (256, 896), 1), (1, (256, 4, 64), (256, 896), 1), (1, (256, 4, 64), (256, 896), 1), (1, (256, 4, 64), (256, 896), 1), (1, (256, 4, 64), (256, 896), 1), (3, 256, 512), (1, (512, 8, 64), (512, 1792), 1), (1, (512, 8, 64), (512, 1792), 1), (1, (512, 8, 64), (512, 1792), 1), (1, (512, 8, 64), (512, 1792), 1), (1, (512, 8, 64), (512, 1792), 1), (1, (512, 8, 64), (512, 1792), 1), (3, 512, 1024), (1, (1024, 12, 64), (1024, 3584), 1), (1, (1024, 12, 64), (1024, 3584), 1), (1, (1024, 12, 64), (1024, 3584), 1), (1, (1024, 12, 64), (1024, 3584), 1), (1, (1024, 12, 64), (1024, 3584), 1), (1, (1024, 12, 64), (1024, 3584), 1), (2, 1024, 1000))' \
    --search-space 'sr_tiny_666' \
    --constraint-value 1580000000 \
    --search-iter 20 \
    --parent-size 75 \
    --init-popu-size 500 \
    --mutate-size 75 \
    --output_dir models/vit-sr-nas/evolutionary_search/tiny_666/no_distill/single_arch/mac@1.58G/