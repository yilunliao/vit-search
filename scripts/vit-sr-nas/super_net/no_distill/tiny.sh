#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'main.py' | awk '{print $2}')

IMAGENET_PATH="/code/imagenet"


python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --model flexible_vit_sr_patch14_224_supernet \
    --batch-size 128 \
    --data-path $IMAGENET_PATH \
    --epochs 120 \
    --no-model-ema \
    --num_workers 8 \
    --output_dir models/vit-sr-nas/super_net/tiny_666/no_distill/sub-train-val/single_arch/ \
    --val-bs 192 \
    --network-def '((0, 256), (1, (256, 4, 64), (256, 768), 1), (1, (256, 4, 64), (256, 768), 1), (1, (256, 4, 64), (256, 768), 1), (1, (256, 4, 64), (256, 768), 1), (1, (256, 4, 64), (256, 768), 1), (1, (256, 4, 64), (256, 768), 1), (3, 256, 512), (1, (512, 8, 64), (512, 1536), 1), (1, (512, 8, 64), (512, 1536), 1), (1, (512, 8, 64), (512, 1536), 1), (1, (512, 8, 64), (512, 1536), 1), (1, (512, 8, 64), (512, 1536), 1), (1, (512, 8, 64), (512, 1536), 1), (3, 512, 1024), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (2, 1024, 1000))' \
    --search-space 'sr_tiny_666' \
    --example-per-arch 64 \
    --single-arch \
    --use-holdout \
    --no-repeated-aug