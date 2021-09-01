import numpy as np
import copy


'''
    _network_def = ((0, 256), 
        (1, (256, 4, 64), (256, 768), 1),
        (1, (256, 4, 64), (256, 768), 1),
        (1, (256, 4, 64), (256, 768), 1),
        (1, (256, 4, 64), (256, 768), 1),
        (1, (256, 4, 64), (256, 768), 1),
        (1, (256, 4, 64), (256, 768), 1),
        (3, 256, 512), 
        (1, (512, 8, 64), (512, 1536), 1),
        (1, (512, 8, 64), (512, 1536), 1),
        (1, (512, 8, 64), (512, 1536), 1),
        (1, (512, 8, 64), (512, 1536), 1),
        (1, (512, 8, 64), (512, 1536), 1),
        (1, (512, 8, 64), (512, 1536), 1),
        (3, 512, 1024), 
        (1, (1024, 12, 64), (1024, 3072), 1),
        (1, (1024, 12, 64), (1024, 3072), 1),
        (1, (1024, 12, 64), (1024, 3072), 1),
        (1, (1024, 12, 64), (1024, 3072), 1),
        (1, (1024, 12, 64), (1024, 3072), 1),
        (1, (1024, 12, 64), (1024, 3072), 1),
        (2, 1024, 1000))
'''


num_channels_to_keep = []

# stage 1
embed = np.array([256, 224, 192, 176, 160])
block = {'attn': np.array([256, 192, 128]), 'mlp': np.array([768, 704, 640, 576, 512, 448, 384]), 'layer': None}  
block_skip = copy.deepcopy(block)
block_skip['layer'] = np.array([256, 256, 0, 0])

num_channels_to_keep.append(embed)
for i in range(3):
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip)
    
# stage 2
embed = np.array([512, 448, 384, 352, 320])
block = {'attn': np.array([512, 384, 256]), 'mlp': np.array([1536, 1408, 1280, 1152, 1024, 896, 768]), 'layer': None}  
block_skip = copy.deepcopy(block)
block_skip['layer'] = np.array([512, 512, 0, 0])

num_channels_to_keep.append(embed)
for i in range(3):
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip)
    
# stage 3
embed = np.array([1024, 896, 768, 704, 640])
block = {'attn': np.array([768, 640, 512, 384]), 'mlp': np.array([3072, 2816, 2560, 2304, 2048, 1792, 1536]), 'layer': None}  
block_skip = copy.deepcopy(block)
block_skip['layer'] = np.array([1024, 1024, 0, 0])

num_channels_to_keep.append(embed)
for i in range(3):
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip)
    
num_channels_to_keep.append(None)