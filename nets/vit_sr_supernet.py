'''
    ViT with spatial reduction
'''

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model

from .supernet_blocks import Block
from .masked_layer_norm import MaskedLayerNorm
from .channel_drop import ChannelDrop
from .patch_conv import PatchConvEmbed


# enum for network_def
_BLOCK_EMBED_INDEX = 0
_EMBED_CHANNEL = 1
_EMBED_CONV_MID_CHANNELS = 2
_BLOCK_HEAD_INDEX = -1
_HEAD_OUT_CHANNEL = 2
_HEAD_IN_CHANNEL  = 1

_BLOCK_TYPE = 0
_TYPE_IS_EMBED = 0
_TYPE_IS_TRANS = 1      # transformer block
_TYPE_IS_HEAD = 2
_TYPE_IS_SR = 3         # spatial reduction patch embedding
_TYPE_IS_CONV_EMBED = 4 # Use several conv for patch embedding
_TYPE_IS_FLEXIBLE_CONV_EMBED = 5 # use different channels for conv patch embedding

_BLOCK_ATTN_IDX = 1
_BLOCK_FFN_IDX  = 2
_BLOCK_EXISTS_IDX = 3
_ATTN_EMBED = 0
_ATTN_NUM_HEAD = 1
_ATTN_HEAD_DIM = 2
_FFN_EMBED = 0
_FFN_HIDDEN = 1

_SR_IN_CHANNEL = 1
_SR_OUT_CHANNEL = 2

_NUM_WARMUP_EPOCHS = 15


class BypassBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, x, embed_mask=None, layer_mask=None):
        layer_mask = None
        return x, embed_mask, layer_mask


class SpatialReductionPatchEmbedding(nn.Module):
    def __init__(self, img_size, in_features, out_features, 
                 patch_size=2, distill_token=True, 
                 num_channels_to_keep=None, num_warmup_epochs=_NUM_WARMUP_EPOCHS,
                 example_per_arch=None, single_arch=False):
        '''
            Input argument:
                `img_size`: the size of input image. The input sequence is reshaped to an image 
                and then split into smaller patches.
        '''
        super(SpatialReductionPatchEmbedding, self).__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.distill_token = distill_token
        if self.distill_token:
            self.num_tokens = 2
        else:
            self.num_tokens = 1

        # norm -> conv
        self.norm = MaskedLayerNorm(num_channels=in_features)
        self.patch_reduce = nn.Conv2d(in_features, out_features, 
            kernel_size=(self.patch_size[0]+1), stride=self.patch_size[0], 
            padding=(self.patch_size[0]//2))
        
        self.patch_pool = nn.AvgPool2d(kernel_size=self.patch_size[0], 
            stride=self.patch_size[0])
        
        # cls token
        assert out_features >= in_features
        #self.token_norm = MaskedLayerNorm(num_channels=in_features)
        self.token_transform = nn.Linear(in_features, out_features)
        
        # for residual path
        self.zero_tensor = torch.zeros(1, 1, out_features - in_features).cuda()
        
        # add position embedding after spatial reduction
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, out_features))
        trunc_normal_(self.pos_embed, std=.02)
        
        # for super-network 
        self.channel_drop = None
        if num_channels_to_keep is not None:
            self.channel_drop = ChannelDrop(num_channels_to_keep=num_channels_to_keep, 
                 num_warmup_epochs=num_warmup_epochs,
                 example_per_arch=example_per_arch, 
                 single_arch=single_arch)
        
    
    def forward(self, x, embed_mask=None, layer_mask=None):
        '''
            1. Split an input sequence into (class + distill) tokens (`cls_tokens`)
            and a sequence of patch embedding.
            2. Patch embedding: 
                (conv_path): layer norm -> conv -> pos_embed
                (residual_path): avg_pool -> concat zero
            3. Class token:
                (transform_path): layer norm -> linear
                (residual_path): concat zero            
        '''
        patch_embed = x[:, self.num_tokens::, :]
        B, N, C = patch_embed.shape
        patch_embed_residual = x[:, self.num_tokens::, :]
        cls_tokens_residual  = x[:, 0:self.num_tokens, :]
        x = self.norm(x, embed_mask)
        
        # patch residual
        B, N, C = patch_embed_residual.shape
        patch_embed_residual = patch_embed_residual.transpose(1, 2)
        patch_embed_residual = patch_embed_residual.reshape(B, C, self.img_size[0], self.img_size[1])
        patch_embed_residual = self.patch_pool(patch_embed_residual)
        patch_embed_residual = patch_embed_residual.flatten(2).transpose(1, 2)
        
        # patch conv
        patch_embed = x[:, self.num_tokens::, :] #self.patch_norm(patch_embed, embed_mask)
        patch_embed = patch_embed.transpose(1, 2)
        patch_embed = patch_embed.reshape(B, C, self.img_size[0], self.img_size[1])
        patch_embed = self.patch_reduce(patch_embed)
        patch_embed = patch_embed.flatten(2).transpose(1, 2)
        patch_embed = patch_embed + self.pos_embed
        
        # token
        #cls_tokens = x[:, 0:self.num_tokens, :]
        #cls_tokens_residual = cls_tokens
        
        #cls_tokens = self.token_norm(cls_tokens, embed_mask)
        cls_tokens = x[:, 0:self.num_tokens, :]
        cls_tokens = self.token_transform(cls_tokens)
        
        # residual
        x_residual = torch.cat((cls_tokens_residual, patch_embed_residual), dim=1)
        B, N, _ = x_residual.shape
        zero_pad_tensor = self.zero_tensor.expand(B, N, -1)
        x_residual = torch.cat((x_residual, zero_pad_tensor), dim=2)
        
        x = torch.cat((cls_tokens, patch_embed), dim=1)
        x = x + x_residual
        
        embed_mask = None
        layer_mask = None
        if self.channel_drop:
            x, embed_mask = self.channel_drop(x)
            
            # debug
            #mask_values = embed_mask.float()
            #mask_values = torch.sum(mask_values, dim=2)
            #print('SR Block:\t', mask_values[0])
        return x, embed_mask, layer_mask
        
    
    def extra_repr(self):
        output_str = super(SpatialReductionPatchEmbedding, self).extra_repr()
        output_str = output_str + 'num_patches={}, distill_token={}, '.format(self.num_patches, self.distill_token)
        if hasattr(self, 'pos_embed'):
            output_str = output_str + 'pos_embed: {}, '.format(self.pos_embed.shape)
        if hasattr(self, 'cls_tokens'):
            output_str = output_str + 'cls_tokens: {}, '.format(self.cls_tokens.shape)
        return output_str
    

class FlexibleDistillVisionTransformerSR(nn.Module):
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
        
        - Add spatial reduction (SR)
        
        - Add `patch_output` for ShiftTokenMixup
        
    """
    def __init__(self, img_size=224, patch_size=14, in_chans=3, num_classes=1000, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=MaskedLayerNorm, distill_token=True, network_def=None, 
                 supernet=False, num_channels_to_keep=None, example_per_arch=None, 
                 num_warmup_epochs=_NUM_WARMUP_EPOCHS, single_arch=False, 
                 hybrid_arch=False, 
                 patch_output=False):
        super().__init__()
        
        assert patch_size == 14
        
        self.network_def = network_def
        
        self.num_classes = num_classes
        assert network_def[_BLOCK_HEAD_INDEX][_HEAD_OUT_CHANNEL] == num_classes
        embed_dim = network_def[_BLOCK_EMBED_INDEX][_EMBED_CHANNEL]
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # Remove hybrid
        patch_embed_class = None
        if network_def[_BLOCK_EMBED_INDEX][_BLOCK_TYPE] in [_TYPE_IS_FLEXIBLE_CONV_EMBED, _TYPE_IS_CONV_EMBED]:
            patch_embed_class = PatchConvEmbed
        else:
            patch_embed_class = PatchEmbed
            
        if network_def[_BLOCK_EMBED_INDEX][_BLOCK_TYPE] == _TYPE_IS_FLEXIBLE_CONV_EMBED:
            mid_chans = network_def[_BLOCK_EMBED_INDEX][_EMBED_CONV_MID_CHANNELS]
            self.patch_embed = patch_embed_class(img_size=img_size, patch_size=patch_size, 
                in_chans=in_chans, embed_dim=embed_dim, mid_chans=mid_chans)
        else:
            self.patch_embed = patch_embed_class(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        img_size = img_size // patch_size
        #self.patch_norm = MaskedLayerNorm(num_channels=embed_dim)
        
        # Remove cls.token. Use tokens instead
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.distill_token = distill_token
        self.num_tokens = 2 if distill_token else 1
        self.tokens = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.patch_output = patch_output

        # super-network related
        self.embed_channel_drop = None
        if supernet:
            assert num_channels_to_keep is not None, 'Super-network numbers of channels to keep error'
            assert (example_per_arch is not None) or single_arch, 'Super-network forward-backward architecture error'
            assert isinstance(num_channels_to_keep, list), 'Num of channels to keep type error'
            assert len(num_channels_to_keep) == len(network_def), 'Lengths of num_channels_to_keep and network_def are not the same'
            self.embed_channel_drop = ChannelDrop(num_channels_to_keep=num_channels_to_keep[0], 
                 num_warmup_epochs=num_warmup_epochs,
                 example_per_arch=example_per_arch, 
                 single_arch=(single_arch or hybrid_arch))

        # construct transform blocks
        depth = 0
        for i in range(len(network_def)):
            if network_def[i][_BLOCK_TYPE] == _TYPE_IS_TRANS:
                depth = depth + 1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
                
        self.blocks = []
        depth = 0
        
        for i in range(len(network_def)):
            block_def = network_def[i]
            if block_def[_BLOCK_TYPE] != _TYPE_IS_SR and block_def[_BLOCK_TYPE] != _TYPE_IS_TRANS:
                continue 
            
            num_chs_keep_block = None
            num_chs_keep_attn  = None
            num_chs_keep_mlp   = None
            num_chs_keep_layer = None
            if supernet:
                num_chs_keep_block = num_channels_to_keep[i]
                if block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
                    assert isinstance(num_chs_keep_block, dict)
                else:
                    assert isinstance(num_chs_keep_block, np.ndarray)
                if isinstance(num_chs_keep_block, dict):
                    num_chs_keep_attn  = num_chs_keep_block['attn']
                    num_chs_keep_mlp   = num_chs_keep_block['mlp']
                    num_chs_keep_layer = num_chs_keep_block['layer']     
            
            if block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
                assert block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED] == block_def[_BLOCK_FFN_IDX][_FFN_EMBED], 'Block {}: embedding dim mismatch'.format(depth)
                assert block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED] == embed_dim, 'Block {}: embedding dim is not consistent with patch embedding'.format(depth)
                if block_def[_BLOCK_EXISTS_IDX]:
                    block_class = Block
                else:
                    block_class = BypassBlock # dummy
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
                
            elif network_def[i][_BLOCK_TYPE] == _TYPE_IS_SR:
                assert block_def[_SR_IN_CHANNEL] == embed_dim, "Block {}: SR input embedding size error".format(i)
                
                # Use fixed patch size
                self.blocks.append(SpatialReductionPatchEmbedding(img_size=img_size, 
                    in_features=block_def[_SR_IN_CHANNEL], 
                    out_features=block_def[_SR_OUT_CHANNEL],
                    num_channels_to_keep=num_chs_keep_block, 
                    num_warmup_epochs=num_warmup_epochs, 
                    example_per_arch=example_per_arch, 
                    single_arch=(single_arch or hybrid_arch), 
                    distill_token=distill_token)
                    )
                
                # update embedding size, image size
                embed_dim = block_def[_SR_OUT_CHANNEL]
                img_size = self.blocks[-1].img_size[0] // self.blocks[-1].patch_size[0]
                    
        self.blocks = nn.ModuleList(self.blocks)

        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        assert embed_dim == network_def[_BLOCK_HEAD_INDEX][_HEAD_IN_CHANNEL]
        self.cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        if self.distill_token:
            self.dst_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.dst_head = None
        
        # ShiftToeknMixup
        #self.patch_norm = None
        self.patch_head = None
        if self.patch_output:
            assert not self.distill_token, 'Currently support only either ShiftTokenMixup or Distillation.'
            #self.patch_norm = norm_layer(embed_dim)
            self.patch_head = nn.Linear(embed_dim, num_classes)
        
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
        #elif isinstance(m, SpatialReductionPatchEmbedding):
        #    trunc_normal_(m.patch_reduce.weight, std=.02)
        #    if m.patch_reduce.bias is not None:
        #        nn.init.constant_(m.patch_reduce.bias, 0)
        
            
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = ['tokens']
        for name, _ in self.blocks.named_parameters():
            if name.endswith(tuple(no_wd_list)):
                no_wd_list.append(name)
        return set(no_wd_list)
                

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
            # debug
            #mask_values = embed_mask.float()
            #mask_values = torch.sum(mask_values, dim=2)
            #print('Patch Embedding:\t', mask_values[0])
        #x = self.patch_norm(x, embed_mask)
        
        #if embed_mask is not None:
        #    x = x *embed_mask        
        
        for blk in self.blocks:
            x, embed_mask, layer_mask = blk(x, embed_mask, layer_mask)
        
        if self.training and self.patch_output:
            x = self.norm(x, embed_mask)
            token_features = x[:, 0:self.num_tokens]
            patch_features = x[:, self.num_tokens::]
            return token_features, patch_features
        else:
            token_features = x[:, 0:self.num_tokens]
            token_features = self.norm(token_features, embed_mask)
            return token_features, None
        #x = self.norm(x, embed_mask)
        #return x[:, 0:self.num_tokens]

    def forward(self, x, patch_output_type=None):
        '''
            When using Patch Token Mixup, 
            output both (classification and patch) prediction for training and
            output only classification prediction for inference.            
        '''
        
        token_features, patch_features = self.forward_features(x)
        cls_pred = self.cls_head(token_features[:, 0])
        
        # for shifted patch token mixup
        if self.patch_output:
            if self.training:
                if patch_output_type == 'seq' or patch_output_type is None:
                    patch_pred = self.patch_head(patch_features)
                elif patch_output_type == 'avg':
                    patch_features = patch_features.mean(dim=1)
                    patch_pred = self.patch_head(patch_features)
                else:
                    raise ValueError()
                return cls_pred, patch_pred
            else:
                return cls_pred
        
        if self.num_tokens == 1:
            return cls_pred
        elif self.num_tokens == 2:
            dst_pred = self.dst_head(token_features[:, 1])
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
def flexible_vit_sr_distill_patch14_224(pretrained=False, **kwargs):
    '''
        Insert `network_def` to **kwargs in main.py
    '''
    model = FlexibleDistillVisionTransformerSR(patch_size=14, distill_token=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_sr_patch14_224(pretrained=False, **kwargs):
    '''
        Insert `network_def` to **kwargs in main.py
    '''
    model = FlexibleDistillVisionTransformerSR(patch_size=14, distill_token=False, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_sr_distill_patch14_224_supernet(pretrained=False, **kwargs):
    '''
        Extra arguments:
            `network_def`, `num_channels_to_keep`, `example_per_arch`, 
            `num_warmup_epochs`, `single_arch=False`
    '''
    model = FlexibleDistillVisionTransformerSR(patch_size=14, distill_token=True, 
        supernet=True,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_sr_patch14_224_supernet(pretrained=False, **kwargs):
    '''
        Extra arguments:
            `network_def`, `num_channels_to_keep`, `example_per_arch`, 
            `num_warmup_epochs`, `single_arch=False`
    '''
    model = FlexibleDistillVisionTransformerSR(patch_size=14, distill_token=False, 
        supernet=True,
        **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_sr_patch14_224_patch_output(pretrained=False, **kwargs):
    '''
        Insert `network_def` to **kwargs in main.py
    '''
    model = FlexibleDistillVisionTransformerSR(patch_size=14, distill_token=False, 
        patch_output=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_sr_patch14_224_patch_output_supernet(pretrained=False, **kwargs):
    '''
        Extra arguments:
            `network_def`, `num_channels_to_keep`, `example_per_arch`, 
            `num_warmup_epochs`, `single_arch=False`
    '''
    model = FlexibleDistillVisionTransformerSR(patch_size=14, distill_token=False, 
        supernet=True, patch_output=True,
        **kwargs)
    model.default_cfg = _cfg()
    return model


'''
    For finetuning at higher resolution (224 + 56 * n)
'''
@register_model
def flexible_vit_sr_patch14_280_patch_output(pretrained=False, **kwargs):
    model = FlexibleDistillVisionTransformerSR(patch_size=14, img_size=280, 
        distill_token=False, patch_output=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_sr_patch14_336_patch_output(pretrained=False, **kwargs):
    model = FlexibleDistillVisionTransformerSR(patch_size=14, img_size=336, 
        distill_token=False, patch_output=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def flexible_vit_sr_patch14_392_patch_output(pretrained=False, **kwargs):
    model = FlexibleDistillVisionTransformerSR(patch_size=14, img_size=392, 
        distill_token=False, patch_output=True, **kwargs)
    model.default_cfg = _cfg()
    return model
