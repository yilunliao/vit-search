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
    --output_dir models/vit-sr-nas/super_net/tiny_conv-patch/example_per_arch@64/ \
    --val-bs 192 \
    --network-def '((4, 256), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (3, 256, 512), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (3, 512, 1024), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (2, 1024, 1000))' \
    --search-space 'sr_tiny_mh' \
    --example-per-arch 64 \
    --use-holdout \
    --no-repeated-aug \
    --use-patch-mixup \
    --drop-path 0.2