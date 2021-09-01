import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, _cfg
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model

from .supernet_blocks import Block
from .masked_layer_norm import MaskedLayerNorm
from .channel_drop import ChannelDrop


# enum for network_def
_BLOCK_EMBED_INDEX = 0
_EMBED_CHANNEL = 1
_BLOCK_HEAD_INDEX = -1
_HEAD_CHANNEL  = 2

_BLOCK_TYPE = 0
_TYPE_IS_EMBED = 0
_TYPE_IS_TRANS = 1 # transformer block
_TYPE_IS_HEAD = 2

_BLOCK_ATTN_IDX = 1
_BLOCK_FFN_IDX  = 2
_BLOCK_EXISTS_IDX = 3
_ATTN_EMBED = 0
_ATTN_NUM_HEAD = 1
_ATTN_HEAD_DIM = 2
_FFN_EMBED = 0
_FFN_HIDDEN = 1


_NUM_WARMUP_EPOCHS = 15


class BypassBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x, embed_mask=None, layer_mask=None):
        return x, embed_mask, layer_mask
    

class FlexibleDistillVisionTransformer(nn.Module):
    """ 
        - Add knowledge distillation to Vision Transformer (DeiT)
        - Remove hybrid 
        - Add argument `distill_token` to distill a pre-trained CNN
        - Rename `self.cls_token` as `self.tokens`
        - Rename `self.head` as `self.cls_head` and add `self.dst_head`
        - Take `network_def` as input so that we can have heterogeneous structures
        - Always use `MaskedLayerNorm`
        - `qkv_scale` and `qkv_bias` are fixed
        
        - `supernet`, `num_channels_to_keep`, `example_per_arch`, 
        `num_warmup_epochs`, and `single_arch` are for super-network training.
        
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=MaskedLayerNorm, distill_token=True, network_def=None, 
                 supernet=False, num_channels_to_keep=None, example_per_arch=None, 
                 num_warmup_epochs=_NUM_WARMUP_EPOCHS, single_arch=False):
        super().__init__()
        
        self.network_def = network_def
        
        self.num_classes = num_classes
        assert network_def[_BLOCK_HEAD_INDEX][_HEAD_CHANNEL] == num_classes
        embed_dim = network_def[_BLOCK_EMBED_INDEX][_EMBED_CHANNEL]
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # Remove hybrid
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Remove cls.token. Use tokens instead
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_tokens = 2 if distill_token else 1
        self.tokens = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # super-network related
        self.embed_channel_drop = None
        if supernet:
            assert num_channels_to_keep is not None, 'Super-network numbers of channels to keep error'
            assert (example_per_arch is not None) or single_arch, 'Super-network forward-backward architecture error'
            assert isinstance(num_channels_to_keep, list), 'Num of channels to keep type error'
            self.embed_channel_drop = ChannelDrop(num_channels_to_keep=num_channels_to_keep[0], 
                 num_warmup_epochs=num_warmup_epochs,
                 example_per_arch=example_per_arch, 
                 single_arch=single_arch)

        # construct transform blocks
        depth = 0
        for i in range(len(network_def)):
            if network_def[i][_BLOCK_TYPE] == _TYPE_IS_TRANS:
                depth = depth + 1
        assert depth == len(network_def) - 2, 'Block number error'
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
                
        self.blocks = []
        depth = 0
        for i in range(len(network_def)):
            if network_def[i][_BLOCK_TYPE] == _TYPE_IS_TRANS:
                block_def = network_def[i]
                assert block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED] == block_def[_BLOCK_FFN_IDX][_FFN_EMBED], 'Block {}: embedding dim mismatch'.format(depth)
                assert block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED] == embed_dim, 'Block {}: embedding dim is not consistent with patch embedding'.format(depth)
                if block_def[_BLOCK_EXISTS_IDX]:
                    block_class = Block
                else:
                    block_class = BypassBlock # dummy 
                    
                if supernet:
                    num_chs_keep_block = num_channels_to_keep[i]
                    num_chs_keep_attn  = num_chs_keep_block['attn']
                    num_chs_keep_mlp   = num_chs_keep_block['mlp']
                    num_chs_keep_layer = num_chs_keep_block['layer'] 
                else:
                    num_chs_keep_attn  = None
                    num_chs_keep_mlp   = None
                    num_chs_keep_layer = None
                
                self.blocks.append(block_class(dim=embed_dim, 
                    num_heads=block_def[_BLOCK_ATTN_IDX][_ATTN_NUM_HEAD], 
                    head_dim=block_def[_BLOCK_ATTN_IDX][_ATTN_HEAD_DIM], 
                    mlp_features=block_def[_BLOCK_FFN_IDX][_FFN_HIDDEN], 
                    drop_path=dpr[depth], 
                    num_chs_to_keep_attn=num_chs_keep_attn, 
                    num_chs_to_keep_mlp=num_chs_keep_mlp, 
                    num_chs_to_keep_block=num_chs_keep_layer,
                    num_warmup_epochs=num_warmup_epochs, 
                    example_per_arch=example_per_arch, 
                    single_arch=single_arch)
                    )
                depth = depth + 1
                
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.dst_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.tokens, std=.02)
        self.apply(self._init_weights)
        
        # super-network related
        self.num_warmup_epochs = num_warmup_epochs
        self.epoch_now = None
        self.is_supernet = supernet
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, MaskedLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'tokens'}

    def get_classifier(self):
        return self.cls_head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.cls_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.dst_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        tokens = self.tokens.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        layer_mask = None
        embed_mask = None
        
        if self.embed_channel_drop is not None:
            x, embed_mask = self.embed_channel_drop.forward(x)
            #print('Embed:', embed_mask)
    
        for blk in self.blocks:
            x, embed_mask, layer_mask = blk(x, embed_mask, layer_mask)

        x = self.norm(x, embed_mask)
        return x[:, 0:self.num_tokens]

    def forward(self, x):
        x = self.forward_features(x)
        
        # x.shape == B, self.num_tokens, C
        
        cls_pred = self.cls_head(x[:, 0])
        
        if self.num_tokens == 1:
            return cls_pred
        elif self.num_tokens == 2:
            dst_pred = self.dst_head(x[:, 1])
            return cls_pred, dst_pred
        else:
            raise ValueError()
    
    
    def set_epoch(self, epoch):
        self.epoch_now = epoch
        
        for m in self.modules():
            if isinstance(m, ChannelDrop):
                m.set_epoch(epoch)
        
        # channel sorting
        if self.is_supernet:
            if self.num_warmup_epochs >= self.epoch_now:
                for m in self.modules():
                    if isinstance(m, Block):
                        m.rewiring()
                

@register_model
def flexible_vit_patch16_224(pretrained=False, **kwargs):
    '''
        Insert `network_def` to **kwargs in main.py
    '''
    model = FlexibleDistillVisionTransformer(patch_size=16, distill_token=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_patch16_224_supernet(pretrained=False, **kwargs):
    '''
        Extra arguments:
            `network_def`, `num_channels_to_keep`, `example_per_arch`, 
            `num_warmup_epochs`, `single_arch=False`
    '''
    model = FlexibleDistillVisionTransformer(patch_size=16, distill_token=True, 
        supernet=True,
        **kwargs)
    model.default_cfg = _cfg()
    return model

   
@register_model
def flexible_vit_patch16_192(pretrained=False, **kwargs):
    '''
        Insert `network_def` to **kwargs in main.py
    '''
    model = FlexibleDistillVisionTransformer(patch_size=16, distill_token=True, 
        img_size=192, 
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_patch16_192_supernet(pretrained=False, **kwargs):
    '''
        Extra arguments:
            `network_def`, `num_channels_to_keep`, `example_per_arch`, 
            `num_warmup_epochs`, `single_arch=False`
    '''
    model = FlexibleDistillVisionTransformer(patch_size=16, distill_token=True, 
        supernet=True,
        img_size=192,
        **kwargs)
    model.default_cfg = _cfg()
    return model