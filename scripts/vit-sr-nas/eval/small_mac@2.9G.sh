#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'main.py' | awk '{print $2}')

IMAGENET_PATH="/data/datasets/imagenet-fast/data/imagenet"
MODEL_PATH="models/vit-sr-nas/searched_net/small/example_per_arch@64/mac@2.9G/epoch@289_checkpoint.pth.tar"


# Also need to change `network_def` for different networks
python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env main.py \
    --eval \
    --model flexible_vit_sr_patch14_224_patch_output \
    --batch-size 128 \
    --data-path $IMAGENET_PATH \
    --num_workers 8 \
    --val-bs 192 \
    --network-def '((4, 220), (1, (220, 5, 32), (220, 880), 1), (1, (220, 5, 32), (220, 880), 1), (1, (220, 7, 32), (220, 800), 1), (1, (220, 5, 32), (220, 720), 1), (1, (220, 5, 32), (220, 720), 1), (1, (220, 5, 32), (220, 720), 1), (3, 220, 440), (1, (440, 10, 48), (440, 1760), 1), (1, (440, 10, 48), (440, 1440), 1), (1, (440, 10, 48), (440, 1920), 1), (1, (440, 10, 48), (440, 1600), 1), (1, (440, 12, 48), (440, 1600), 1), (1, (440, 12, 48), (440, 1440), 1), (3, 440, 880), (1, (880, 16, 64), (880, 3200), 1), (1, (880, 12, 64), (880, 3200), 1), (1, (880, 16, 64), (880, 2880), 1), (1, (880, 12, 64), (880, 2240), 1), (1, (880, 14, 64), (880, 2560), 1), (2, 880, 1000))' \
    --resume $MODEL_PATH \
    --no-model-ema