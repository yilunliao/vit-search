import numpy as np
import copy


'''
    _network_def = ((4, 320), 
        (1, (320, 8, 32), (320, 960), 1),
        (1, (320, 8, 32), (320, 960), 1),
        (1, (320, 8, 32), (320, 960), 1),
        (1, (320, 8, 32), (320, 960), 1),
        (1, (320, 8, 32), (320, 960), 1),
        (1, (320, 8, 32), (320, 960), 1),
        (1, (320, 8, 32), (320, 960), 1),
        (3, 320, 640), 
        (1, (640, 16, 48), (640, 1920), 1),
        (1, (640, 16, 48), (640, 1920), 1),
        (1, (640, 16, 48), (640, 1920), 1),
        (1, (640, 16, 48), (640, 1920), 1),
        (1, (640, 16, 48), (640, 1920), 1),
        (1, (640, 16, 48), (640, 1920), 1),
        (1, (640, 16, 48), (640, 1920), 1),
        (3, 640, 1280), 
        (1, (1280, 16, 64), (1280, 3840), 1),
        (1, (1280, 16, 64), (1280, 3840), 1),
        (1, (1280, 16, 64), (1280, 3840), 1),
        (1, (1280, 16, 64), (1280, 3840), 1),
        (1, (1280, 16, 64), (1280, 3840), 1),
        (1, (1280, 16, 64), (1280, 3840), 1),
        (1, (1280, 16, 64), (1280, 3840), 1),
        (2, 1280, 1000))
'''


num_channels_to_keep = []

# stage 1
embed = np.array([320, 280, 240, 220, 200])
block = {'attn': np.array([256, 224, 192, 160]), 'mlp': np.array([960, 880, 800, 720, 640, 560, 480]), 'layer': None}  
block_skip = copy.deepcopy(block)
block_skip['layer'] = np.array([320, 320, 0, 0])

num_channels_to_keep.append(embed)
for i in range(3):
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip)
num_channels_to_keep.append(block)

# stage 2
embed = np.array([640, 560, 480, 440, 400])
block = {'attn': np.array([768, 672, 576, 480]), 'mlp': np.array([1920, 1760, 1600, 1440, 1280, 1120, 960]), 'layer': None}  
block_skip = copy.deepcopy(block)
block_skip['layer'] = np.array([640, 640, 0, 0])

num_channels_to_keep.append(embed)
for i in range(3):
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip)
num_channels_to_keep.append(block)

# stage 3
embed = np.array([1280, 1120, 960, 880, 800])
block = {'attn': np.array([1024, 896, 768, 640]), 'mlp': np.array([3840, 3520, 3200, 2880, 2560, 2240, 1920]), 'layer': None}  
block_skip = copy.deepcopy(block)
block_skip['layer'] = np.array([1280, 1280, 0, 0])

num_channels_to_keep.append(embed)
for i in range(3):
    num_channels_to_keep.append(block)
    num_channels_to_keep.append(block_skip)
num_channels_to_keep.append(block)
    
num_channels_to_keep.append(None)