#!/bin/bash
#ulimit -n 4096
#NODE_RANK=${1}

kill $(ps aux | grep 'evo_search.py' | awk '{print $2}')

IMAGENET_PATH="/code/imagenet"
MODEL_PATH="models/vit-sr-nas/super_net/tiny_conv-patch/example_per_arch@64/epoch@119_checkpoint.pth.tar"

python -m torch.distributed.launch --nproc_per_node=8 \
    --use_env evo_search.py \
    --data-path $IMAGENET_PATH \
    --val-bs 256 \
    --num_workers 8 \
    --model-path $MODEL_PATH \
    --model 'flexible_vit_sr_patch14_224_patch_output' \
    --network-def '((4, 256), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (1, (256, 6, 32), (256, 768), 1), (3, 256, 512), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (1, (512, 12, 48), (512, 1536), 1), (3, 512, 1024), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (1, (1024, 12, 64), (1024, 3072), 1), (2, 1024, 1000))' \
    --search-space 'sr_tiny_mh' \
    --constraint-value 1794400000 \
    --search-iter 20 \
    --parent-size 75 \
    --init-popu-size 500 \
    --mutate-size 75 \
    --output_dir models/vit-sr-nas/evolutionary_search/tiny_conv-patch/example_per_arch@64/mac@1.794G
    
  