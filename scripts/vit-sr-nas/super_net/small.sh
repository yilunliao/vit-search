#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'main.py' | awk '{print $2}')

IMAGENET_PATH="/code/imagenet"

python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --model flexible_vit_sr_patch14_224_patch_output_supernet \
    --batch-size 128 \
    --no-model-ema \
    --data-path $IMAGENET_PATH \
    --epochs 120 \
    --num_workers 8 \
    --output_dir models/vit-sr-nas/super_net/small/example_per_arch@64/ \
    --val-bs 192 \
    --network-def '((4, 320), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (1, (320, 8, 32), (320, 960), 1), (3, 320, 640), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1920), 1), (3, 640, 1280), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (1, (1280, 16, 64), (1280, 3840), 1), (2, 1280, 1000))' \
    --search-space 'sr_small_mh' \
    --example-per-arch 64 \
    --use-holdout \
    --no-repeated-aug \
    --use-patch-mixup \
    --drop-path 0.3