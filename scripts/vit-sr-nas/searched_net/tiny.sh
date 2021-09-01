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
    --output_dir models/vit-sr-nas/searched_net/tiny/example_per_arch@64/ \
    --val-bs 192 \
    --network-def '((4, 176), (1, (176, 3, 32), (176, 704), 1), (1, (176, 3, 32), (176, 576), 1), (1, (176, 3, 32), (176, 640), 1), (1, (176, 4, 32), (176, 576), 1), (1, (176, 4, 32), (176, 704), 1), (3, 176, 352), (1, (352, 10, 48), (352, 1408), 1), (1, (352, 8, 48), (352, 1408), 1), (1, (352, 8, 48), (352, 1280), 1), (1, (352, 8, 48), (352, 1408), 1), (1, (352, 10, 48), (352, 1280), 1), (1, (352, 10, 48), (352, 1024), 1), (3, 352, 704), (1, (704, 10, 64), (704, 2560), 1), (1, (704, 10, 64), (704, 1792), 1), (1, (704, 10, 64), (704, 2816), 1), (1, (704, 8, 64), (704, 2816), 1), (1, (704, 8, 64), (704, 2560), 1), (2, 704, 1000))' \
    --no-repeated-aug \
    --use-patch-mixup \
    --drop-path 0.2