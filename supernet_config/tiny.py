import numpy as np
import copy


num_channels_to_keep = []

embed = np.array([240, 224, 208, 192])
block = {'attn': np.array([512, 384, 256, 128]), 'mlp': np.array([1024, 768, 512, 256]), 'layer': None}  
block_skip_1 = copy.deepcopy(block)
block_skip_1['layer'] = np.array([240, 240, 0])
block_skip_2 = copy.deepcopy(block)
block_skip_2['layer'] = np.array([240, 0])

num_channels_to_keep.append(embed)
num_channels_to_keep.append(block)
for i in range(3):
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip_1)
    num_channels_to_keep.append(block_skip_2)
num_channels_to_keep.append(block)
num_channels_to_keep.append(None)
