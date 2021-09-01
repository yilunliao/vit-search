#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'main.py' | awk '{print $2}')

IMAGENET_PATH="/data/datasets/imagenet-fast/data/imagenet"
FINETUNE_PATH="models/vit-sr-nas/searched_net/medium/example_per_arch@64/mac@4.6G/best_ema_checkpoint.pth.tar"
OUTPUT_DIR="models/vit-sr-nas/finetune/medium/input-size@392"

python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --model flexible_vit_sr_patch14_280_patch_output \
    --batch-size 32 \
    --data-path $IMAGENET_PATH \
    --epochs 30 \
    --num_workers 8 \
    --output_dir $OUTPUT_DIR \
    --val-bs 64 \
    --network-def '((4, 240), (1, (240, 7, 32), (240, 960), 1), (1, (240, 6, 32), (240, 960), 1), (1, (240, 7, 32), (240, 800), 1), (1, (240, 8, 32), (240, 960), 1), (1, (240, 7, 32), (240, 880), 1), (1, (240, 8, 32), (240, 880), 1), (1, (240, 6, 32), (240, 800), 1), (3, 240, 640), (1, (640, 10, 48), (640, 1120), 1), (1, (640, 14, 48), (640, 1760), 1), (1, (640, 14, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1760), 1), (1, (640, 14, 48), (640, 1440), 1), (1, (640, 16, 48), (640, 1760), 1), (1, (640, 16, 48), (640, 1920), 1), (3, 640, 880), (1, (880, 16, 64), (880, 3200), 1), (1, (880, 10, 64), (880, 3840), 1), (1, (880, 16, 64), (880, 3840), 1), (1, (880, 12, 64), (880, 3200), 1), (1, (880, 16, 64), (880, 3520), 1), (1, (880, 14, 64), (880, 3520), 1), (2, 880, 1000))' \
    --no-repeated-aug \
    --use-patch-mixup \
    --finetune $FINETUNE_PATH \
    --drop-path 0.75 \
    --input-size 280 \
    --mixup-patch-len 5 \
    --lr 5e-6 \
    --min-lr 5e-6 \
    --weight-decay 1e-8