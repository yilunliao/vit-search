import torch
import torch.nn as nn
import numpy as np
#from timm.layers import DropPath

from .channel_drop import ChannelDrop
from .drop import DropPath
from .masked_layer_norm import MaskedLayerNorm


_NUM_WARMUP_EPOCHS_CHANNEL = 15
_EXAMPLE_PER_ARCH = 16

# ChannelDrop(num_channels_to_keep=None, 
#             num_warmup_epochs=_NUM_WARMUP_EPOCHS_CHANNEL,
#             example_per_arch=None, single_arch=False)
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., 
        num_channels_to_keep=None, num_warmup_epochs=_NUM_WARMUP_EPOCHS_CHANNEL, 
        example_per_arch=_EXAMPLE_PER_ARCH, single_arch=False):
        
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
        self.channel_drop_layer = None
        if num_channels_to_keep is not None:
            self.channel_drop_layer = ChannelDrop(num_channels_to_keep=num_channels_to_keep, 
                num_warmup_epochs=num_warmup_epochs, example_per_arch=example_per_arch, 
                single_arch=single_arch)
            

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        
        if self.channel_drop_layer is not None:
            x, _ = self.channel_drop_layer.forward(x)
            # for debug
            #x, mask = self.channel_drop_layer.forward(x)
            #mask_f = mask.float()
            #mask_f = torch.sum(mask_f, dim=2, keepdim=True)
            #print('mlp:\t', mask_f[0])

        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    
    def rewiring(self):
        
        out_weight = self.fc2.weight.data # shape: (out, hidden)
        out_weight = torch.abs(out_weight)
        out_weight = torch.sum(out_weight, dim=0)
        hidden_weight = self.fc1.weight.data
        hidden_weight = torch.abs(hidden_weight)
        hidden_weight = torch.sum(hidden_weight, dim=1)
        hidden_bias = torch.abs(self.fc1.bias.data)
        weight = out_weight + hidden_weight + hidden_bias
        
        _, hidden_sorted_index = torch.sort(weight, descending=True)
        self.fc1.weight.data = self.fc1.weight.data[hidden_sorted_index, :]
        self.fc1.bias.data = self.fc1.bias.data[hidden_sorted_index]
        self.fc2.weight.data = self.fc2.weight.data[:, hidden_sorted_index]
        
        return 
    
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim=64, 
        qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., 
        num_channels_to_keep=None, num_warmup_epochs=_NUM_WARMUP_EPOCHS_CHANNEL, 
        example_per_arch=_EXAMPLE_PER_ARCH, single_arch=False):
        
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, num_heads * head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(num_heads * head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.channel_drop_layer = None
        if num_channels_to_keep is not None:
            self.channel_drop_layer = ChannelDrop(num_channels_to_keep=num_channels_to_keep, 
                num_warmup_epochs=num_warmup_epochs, example_per_arch=example_per_arch, 
                single_arch=single_arch)
            

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        
        if self.channel_drop_layer is not None:
            x, _ = self.channel_drop_layer(x)
            # for debug
            #x, mask = self.channel_drop_layer(x)
            #mask_f = mask.float()
            #mask_f = torch.sum(mask_f, dim=2, keepdim=True)
            #print('attn:\t', mask_f[0])    
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
    def rewiring(self):
        qkv_weight = self.qkv.weight.data
        qkv_vector = torch.abs(qkv_weight)
        qkv_vector = torch.sum(qkv_vector, dim=1)
        qkv_vector = qkv_vector.reshape((3, self.num_heads, self.head_dim))
        qkv_vector = torch.sum(qkv_vector, dim=0)
        qkv_vector = torch.sum(qkv_vector, dim=1)
        
        qkv_bias = self.qkv.bias.data
        qkv_bias_vector = torch.abs(qkv_bias)
        qkv_bias_vector = qkv_bias_vector.reshape((3, self.num_heads, self.head_dim))
        qkv_bias_vector = torch.sum(qkv_bias_vector, dim=0)
        qkv_bias_vector = torch.sum(qkv_bias_vector, dim=1)
        
        proj_weight = self.proj.weight.data
        proj_vector = torch.abs(proj_weight)
        proj_vector = torch.sum(proj_vector, dim=0)
        proj_vector = proj_vector.reshape((self.num_heads, self.head_dim))
        proj_vector = torch.sum(proj_vector, dim=1)
        
        weight = proj_vector + qkv_bias_vector + qkv_vector
        _, sorted_index = torch.sort(weight, descending=True)
        
        # re-ordering
        qkv_weight = qkv_weight.reshape((3, self.num_heads, self.head_dim, -1))
        qkv_weight = qkv_weight[:, sorted_index, :, :]
        qkv_weight = qkv_weight.reshape(3 * self.num_heads * self.head_dim, -1)
        qkv_bias = qkv_bias.reshape((3, self.num_heads, self.head_dim))
        qkv_bias = qkv_bias[:, sorted_index, :]
        qkv_bias = qkv_bias.reshape(3 * self.num_heads * self.head_dim)
        proj_weight = proj_weight.reshape((-1, self.num_heads, self.head_dim))
        proj_weight = proj_weight[:, sorted_index, :]
        proj_weight = proj_weight.reshape(-1, self.num_heads * self.head_dim)
        
        self.qkv.weight.data = qkv_weight
        self.qkv.bias.data = qkv_bias
        self.proj.weight.data = proj_weight
        
        return
    

class Block(nn.Module):

    def __init__(self, dim, num_heads, head_dim, mlp_features, 
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,  
                 num_chs_to_keep_attn=None, num_chs_to_keep_mlp=None, 
                 num_chs_to_keep_block=None,
                 num_warmup_epochs=_NUM_WARMUP_EPOCHS_CHANNEL, 
                 example_per_arch=_EXAMPLE_PER_ARCH, single_arch=False):
        '''
            1. Add `num_chs_to_keep_attn` and `num_chs_to_keep_mlp` to specify channel drop.
            
            2. Add `num_chs_to_keep_block` to support dropping all layers within the block.
            
            3. `num_warmup_epochs`, `example_per_arch` and `single_arch` are shared 
            across attention and mlp.
            
            4. Use `mlp_features` to specify the number of hidden size in MLP.
            
            5. Always use `MaskedLayerNorm` for normalization
            
        '''
        super().__init__()
        
        # for removing the output of attention and mlp
        self.layer_drop = None
        if num_chs_to_keep_block is not None:
            self.layer_drop = ChannelDrop(num_channels_to_keep=num_chs_to_keep_block, 
                num_warmup_epochs=num_warmup_epochs, example_per_arch=example_per_arch, 
                single_arch=single_arch)
        
        self.norm1 = MaskedLayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, head_dim=head_dim, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
            num_channels_to_keep=num_chs_to_keep_attn, num_warmup_epochs=num_warmup_epochs, 
            example_per_arch=example_per_arch, single_arch=single_arch)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = MaskedLayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_features, act_layer=act_layer, drop=drop, 
            num_channels_to_keep=num_chs_to_keep_mlp, num_warmup_epochs=num_warmup_epochs, 
            example_per_arch=example_per_arch, single_arch=single_arch)
    

    def forward(self, x, embed_mask=None, layer_mask=None):
        '''
            Input to the block is already masked (i.e. the last few channels can be zeroed out.)
        '''
        f_x = self.norm1(x, embed_mask)
        f_x = self.drop_path(self.attn(f_x))
        
        current_layer_mask = None
        
        # Only consider previous layer mask `layer_mask`
        # when the current block has `layer_drop`
        if self.layer_drop is not None:
            f_x, current_layer_mask = self.layer_drop(f_x)
            if layer_mask is not None:
                current_layer_mask = current_layer_mask & layer_mask
            # for debug
            #print('Layer mask:', current_layer_mask)
        else:
            current_layer_mask = None
            
        # for debug
        #if current_layer_mask is not None:
        #    mask_f = current_layer_mask.float()
        #    mask_f = torch.sum(mask_f, dim=2, keepdim=True)
        #    mask_f = mask_f[0] > 0
        #else:
        #    mask_f = current_layer_mask
        #print('Layer:\t', mask_f)            
        
        if embed_mask is not None:
            if current_layer_mask is not None:
                current_layer_mask = current_layer_mask & embed_mask
            else:
                current_layer_mask = embed_mask
            f_x = f_x * current_layer_mask
        
        x = x + f_x

        f_x = self.norm2(x, embed_mask)
        f_x = self.drop_path(self.mlp(f_x))
        
        if current_layer_mask is not None:
            f_x = f_x * current_layer_mask
        
        x = x + f_x
        
        return x, embed_mask, current_layer_mask
    
    
    def rewiring(self):
        self.attn.rewiring()
        self.mlp.rewiring()