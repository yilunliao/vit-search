#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'main.py' | awk '{print $2}')

IMAGENET_PATH="/data/datasets/imagenet-fast/data/imagenet"


python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --model flexible_vit_sr_patch14_224 \
    --batch-size 128 \
    --data-path $IMAGENET_PATH \
    --epochs 300 \
    --no-model-ema \
    --num_workers 12 \
    --output_dir models/vit-sr-nas/searched_net/tiny_666/single_arch/no_distill/mac@1.58G/epochs@300 \
    --val-bs 192 \
    --network-def '((0, 176), (1, (176, 2, 64), (176, 576), 1), (1, (176, 2, 64), (176, 704), 1), (1, (176, 2, 64), (176, 512), 1), (1, (176, 2, 64), (176, 640), 1), (1, (176, 2, 64), (176, 640), 1), (1, (176, 2, 64), (176, 640), 1), (3, 176, 352), (1, (352, 4, 64), (352, 1280), 1), (1, (352, 6, 64), (352, 1024), 1), (1, (352, 8, 64), (352, 1280), 1), (1, (352, 4, 64), (352, 1536), 1), (1, (352, 6, 64), (352, 1408), 1), (3, 352, 704), (1, (704, 8, 64), (704, 2560), 1), (1, (704, 12, 64), (704, 2816), 1), (1, (704, 6, 64), (704, 3072), 1), (1, (704, 8, 64), (704, 2304), 1), (1, (704, 10, 64), (704, 2560), 1), (2, 704, 1000))'