import numpy as np
import copy


num_channels_to_keep = []

embed = np.array([384, 352, 320, 288])
block = {'attn': np.array([512, 448, 384, 320]), 'mlp': np.array([1536, 1280, 1024, 768]), 'layer': None}  
block_skip_1 = copy.deepcopy(block)
block_skip_1['layer'] = np.array([384, 384, 384, 0])
block_skip_2 = copy.deepcopy(block)
block_skip_2['layer'] = np.array([384, 384, 0, 0])

num_channels_to_keep.append(embed)
num_channels_to_keep.append(block)
num_channels_to_keep.append(block)
for i in range(3):
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip_1)
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip_1)
num_channels_to_keep.append(block)
num_channels_to_keep.append(block)
num_channels_to_keep.append(None)
