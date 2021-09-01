import torch
from torch import nn as nn
from torch.nn import functional as F

from timm.models.layers import to_2tuple


#https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/activations.py
def hard_swish(x, inplace: bool = False):
    inner = F.relu6(x + 3.).div_(6.)
    return x.mul_(inner) if inplace else x.mul(inner)


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, self.inplace)
    

class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU() #HardSwish(inplace=True)
    
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    

class PatchConvEmbed(nn.Module):
    '''
        Use several conv for patch embedding
    '''
    def __init__(self, embed_dim, img_size=224, patch_size=14, in_chans=3, mid_chans=24):
        super(PatchConvEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.conv1 = ConvBnAct(in_channels=in_chans, out_channels=mid_chans, stride=(2, 2))
        self.conv2 = ConvBnAct(in_channels=mid_chans, out_channels=mid_chans)
        self.conv3 = ConvBnAct(in_channels=mid_chans, out_channels=mid_chans)
        
        assert self.patch_size[0] % 2 == 0
        assert self.patch_size[1] % 2 == 0
        self.conv_proj = nn.Conv2d(mid_chans, embed_dim, 
            kernel_size=(self.patch_size[0]//2, self.patch_size[1]//2), 
            stride=(self.patch_size[0]//2, self.patch_size[1]//2))
        
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        
        x = self.conv1(x)
        x_res = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + x_res
        x = self.conv_proj(x).flatten(2).transpose(1, 2)
        
        return x