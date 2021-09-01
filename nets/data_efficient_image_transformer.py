'''
    Making some changes to vision transformers to see whether the network architecture can be improved.
    
    1. Replace LN with GN and insert GN into the middle of MLP layers.
'''

import torch
import torch.nn as nn
from functools import partial

from timm.models.registry import register_model
from timm.models.vision_transformer import Attention, PatchEmbed, Mlp, Block, _cfg
from timm.models.layers import to_2tuple, trunc_normal_, drop_path


class DistillVisionTransformer(nn.Module):
    """ 
        - Add knowledge distillation to Vision Transformer (DeiT)
        - Remove hybrid 
        - Add argument `distill_token` to distill a pre-trained CNN
        - Rename `self.cls_token` as `self.tokens`
        - Rename `self.head` as `self.cls_head` and add `self.dst_head`
        
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, distill_token=True):
        super().__init__()
        self.num_classes = num_classes
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

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        #self.repr = nn.Linear(embed_dim, representation_size)
        #self.repr_act = nn.Tanh()

        # Classifier head
        self.cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.dst_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.tokens, std=.02)
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
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

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
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


# Borrowed from DeiT models.py
@register_model
def deit_tiny_distill_patch16_224(pretrained=False, **kwargs):
    model = DistillVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distill_token=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_tiny_133X_distill_patch16_224(pretrained=False, **kwargs):
    model = DistillVisionTransformer(
        patch_size=16, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distill_token=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_tiny_167X_distill_patch16_224(pretrained=False, **kwargs):
    model = DistillVisionTransformer(
        patch_size=16, embed_dim=320, depth=12, num_heads=5, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distill_token=True, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_small_distill_patch16_224(pretrained=False, **kwargs):
    model = DistillVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), distill_token=True, **kwargs)
    model.default_cfg = _cfg()
    return model