'''
   Calculate vision transformer MAC or FLOPs given `network_def`.
   
   Refernce: https://github.com/google-research/electra/blob/master/flops_computation.py
'''

# enum for network_def
_BLOCK_EMBED_INDEX = 0
_EMBED_CHANNEL = 1
_EMBED_CONV_MID_CHANNELS = 2
_BLOCK_HEAD_INDEX = -1
_HEAD_CHANNEL  = 2

_BLOCK_TYPE = 0
_TYPE_IS_EMBED = 0
_TYPE_IS_TRANS = 1 # transformer block
_TYPE_IS_HEAD = 2
_TYPE_IS_SR_BLOCK = 3
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

_SR_BLOCK_NUM_IN_CHANNELS = 1
_SR_BLOCK_NUM_OUT_CHANNELS = 2
    
# for transformer block
_NUM_SEQ = 196 + 2
_SOFTMAX_FLOPS = 5
_LAYER_NORM_FLOPS = 5
_GELU_ACTIVATION_FLOPS = 8

# for patch embedding
_PATCH_SIZE = 16
_NUM_PATCH = 14 * 14
_NUM_CHS = 3

_NUM_CLASSES = 1000


_RESOLUTION_PATCH_DICT = {192: 12 * 12, 224: 14 * 14}
_RESOLUTION_NUM_SEQ_DICT = {192: 12 * 12 + 2, 224: 14 * 14 + 2}


def compute_attention(embed_dim, num_heads, head_dim, n_seq, return_mac=True):
    
    multi_add_factor = 1 if return_mac else 2
    bias_factor = 0 if return_mac else 1
    misc_factor = 0 if return_mac else 1
    
    compute = 0
    
    # embedding -> qkv
    compute = compute + embed_dim * num_heads * head_dim * 3 * n_seq * multi_add_factor
    compute = compute + num_heads * head_dim * 3 * n_seq * bias_factor
    # attention score
    compute = compute + n_seq * n_seq * num_heads * head_dim * multi_add_factor
    compute = compute + n_seq * num_heads * n_seq * _SOFTMAX_FLOPS * misc_factor
    compute = compute + n_seq * n_seq * num_heads * misc_factor # scale
    compute = compute + n_seq * n_seq * head_dim * num_heads * multi_add_factor # weighted average
    compute = compute + n_seq * (head_dim * num_heads * embed_dim) * multi_add_factor
    compute = compute + n_seq * embed_dim * bias_factor
    compute = compute + n_seq * embed_dim * misc_factor # residual add
    compute = compute + n_seq * embed_dim * _LAYER_NORM_FLOPS * misc_factor
    
    return compute


def compute_feedforward_network(embed_dim, hidden_size, n_seq, return_mac=True):
    multi_add_factor = 1 if return_mac else 2
    bias_factor = 0 if return_mac else 1
    misc_factor = 0 if return_mac else 1
    
    compute = 0
    
    compute = compute + n_seq * embed_dim * hidden_size * multi_add_factor
    compute = compute + n_seq * hidden_size * bias_factor
    compute = compute + n_seq * hidden_size * _GELU_ACTIVATION_FLOPS * misc_factor
    
    compute = compute + n_seq * embed_dim * hidden_size * multi_add_factor
    compute = compute + n_seq * embed_dim * bias_factor
    compute = compute + n_seq * embed_dim * misc_factor # residual add
    compute = compute + n_seq * embed_dim * _LAYER_NORM_FLOPS * misc_factor
    
    return compute
    

def compute_transformer_block(block_def, n_seq, return_mac=True):
    '''
        - block_def: e.g. (1, (192, 3, 64), (192, 768), 1)
        - n_seq: length of sequences
        - return_mac: True, return MAC (ignore computation in normalization, softmax...).
            Otherwise, return FLOPs
    '''
    if not block_def[_BLOCK_EXISTS_IDX]:
        return 0
    
    compute = 0
    compute_attn = compute_attention(embed_dim=block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED], 
        num_heads=block_def[_BLOCK_ATTN_IDX][_ATTN_NUM_HEAD], 
        head_dim=block_def[_BLOCK_ATTN_IDX][_ATTN_HEAD_DIM], 
        n_seq=n_seq, return_mac=return_mac)
    compute = compute + compute_attn
    
    compute_ffn = compute_feedforward_network(embed_dim=block_def[_BLOCK_FFN_IDX][_FFN_EMBED], 
        hidden_size=block_def[_BLOCK_FFN_IDX][_FFN_HIDDEN], 
        n_seq=n_seq, return_mac=return_mac)
    compute = compute + compute_ffn
    
    #print('Attn:', compute_attn)
    #print('FFN: ', compute_ffn)
    return compute


def compute_patch_embedding(embed_dim, num_patch, num_chs=_NUM_CHS, patch_size=_PATCH_SIZE, return_mac=True, mid_chs=None, conv_embedding=False):
    multi_add_factor = 1 if return_mac else 2
    bias_factor = 0 if return_mac else 1
    
    compute = 0

    if conv_embedding:
        # assume input size is 3 x 224 x 224
        
        #_mid_chs = 24
        assert mid_chs is not None
        _k = 3
        _mid_resolution = 112
        patch_size = patch_size // 2
        compute = compute + (num_chs * mid_chs * _k * _k) * _mid_resolution * _mid_resolution * multi_add_factor
        compute = compute + (mid_chs * _mid_resolution * _mid_resolution) * bias_factor
        compute = compute + (mid_chs * mid_chs * _k * _k) * _mid_resolution * _mid_resolution * multi_add_factor * 2
        compute = compute + (mid_chs * _mid_resolution * _mid_resolution) * bias_factor * 2
        compute = compute + (embed_dim * mid_chs) * patch_size * patch_size * num_patch * multi_add_factor
        compute = compute + embed_dim * num_patch * bias_factor
    else:
        compute = compute + (embed_dim * num_chs) * patch_size * patch_size * num_patch * multi_add_factor
        compute = compute + embed_dim * num_patch * bias_factor
        
    return compute

    
def compute_position_embedding(embed_dim, n_seq, return_mac=True):
    bias_factor = 0 if return_mac else 1
    return embed_dim * n_seq * bias_factor


def compute_head(embed_dim, n_seq, num_classes=_NUM_CLASSES, return_mac=True):
    multi_add_factor = 1 if return_mac else 2
    bias_factor = 0 if return_mac else 1
    misc_factor = 0 if return_mac else 1
    
    compute = 0
    
    compute = compute + embed_dim * _LAYER_NORM_FLOPS * misc_factor
    compute = compute + embed_dim * num_classes * multi_add_factor
    compute = compute + n_seq * num_classes * bias_factor
    
    return compute


def compute_sr_block(img_size, patch_size, num_in_channels, num_out_channels, distill, return_mac=True):
    multi_add_factor = 1 if return_mac else 2
    bias_factor = 0 if return_mac else 1
    misc_factor = 0 if return_mac else 1
    
    compute = 0
    
    # patch embedding
    assert img_size % patch_size == 0
    output_size = img_size // patch_size
    compute = compute + (output_size * output_size * num_out_channels) * ((patch_size + 1) * (patch_size + 1) * num_in_channels) * multi_add_factor
    compute = compute + output_size * output_size * num_out_channels * bias_factor
    compute = compute + output_size * output_size * num_out_channels * _LAYER_NORM_FLOPS * misc_factor
    compute = compute + output_size * output_size * num_out_channels * bias_factor # position embedding
    
    # token embedding
    token_compute = 0
    token_compute = token_compute + num_in_channels * _LAYER_NORM_FLOPS * misc_factor
    token_compute = token_compute + num_in_channels * num_out_channels * multi_add_factor
    token_compute = token_compute + num_out_channels * bias_factor
    token_compute = token_compute + num_in_channels * misc_factor # residual add
    
    if distill:
        token_compute = token_compute * 2
    compute = compute + token_compute
    return compute
    

def get_compute_from_network_def(network_def, resolution=224, return_mac=True):
    
    compute = 0
    assert network_def[_BLOCK_EMBED_INDEX][_BLOCK_TYPE] == _TYPE_IS_EMBED, "Network def error: embedding"
    assert resolution in _RESOLUTION_NUM_SEQ_DICT.keys()
    
    # for different resolution
    _num_patch = _RESOLUTION_PATCH_DICT[resolution]
    _n_seq = _RESOLUTION_NUM_SEQ_DICT[resolution]
    
    embed_dim = network_def[_BLOCK_EMBED_INDEX][_EMBED_CHANNEL]
    compute = compute + compute_patch_embedding(embed_dim, num_patch=_num_patch, return_mac=return_mac)
    
    compute = compute + compute_position_embedding(embed_dim, n_seq=_n_seq, return_mac=return_mac)
    
    for i in range(len(network_def)):
        block_def = network_def[i]
        if block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
            compute = compute + compute_transformer_block(block_def, n_seq=_n_seq, return_mac=return_mac)
            assert block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED] == block_def[_BLOCK_FFN_IDX][_FFN_EMBED], 'Block {}: embedding dim mismatch'.format(i)
            assert block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED] == embed_dim, 'Block {}: embedding dim is not consistent with patch embedding'.format(i)
    
    num_classes = network_def[_BLOCK_HEAD_INDEX][_HEAD_CHANNEL]
    # assume using both classification and ditillation heads
    compute = compute + compute_head(embed_dim, num_classes=num_classes, n_seq=_n_seq, return_mac=return_mac) * 2
    
    return compute


# Use class to estimate MAC or FLOPs for ViT-SR or ViT
class ComputationEstimator():
    def __init__(self, distill, input_resolution, patch_size, num_in_channels=_NUM_CHS, return_mac=True):
        # patch size can be merged to network_def though
        self.distill = distill # add distillation token
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.num_in_channels = num_in_channels
        self.return_mac = return_mac
        
        assert self.input_resolution % self.patch_size == 0
        
        # use fixed patch size for SR block
        self.sr_patch_size = 2
     
        
    def __repr__(self):
        return '(distill={}, input_resolution={}, patch_size={}, sr_patch_size={}, num_in_channels={}, return_mac={})'.format(self.distill, 
            self.input_resolution, self.patch_size, self.sr_patch_size, 
            self.num_in_channels, self.return_mac)
    
    
    def __call__(self, network_def):
        compute = 0
        
        num_patches = self.input_resolution // self.patch_size
        num_patches = num_patches * num_patches
        n_seq = 2 if self.distill else 1
        n_seq = n_seq + num_patches
        return_mac = self.return_mac
        img_size = self.input_resolution // self.patch_size # for SR block
        
        assert network_def[_BLOCK_EMBED_INDEX][_BLOCK_TYPE] in [_TYPE_IS_EMBED, _TYPE_IS_CONV_EMBED, _TYPE_IS_FLEXIBLE_CONV_EMBED], "Network def error: embedding"
        embed_dim = network_def[_BLOCK_EMBED_INDEX][_EMBED_CHANNEL]
        conv_embedding = False if network_def[_BLOCK_EMBED_INDEX][_BLOCK_TYPE] == _TYPE_IS_EMBED else True
        mid_chs = None
        if network_def[_BLOCK_EMBED_INDEX][_BLOCK_TYPE] == _TYPE_IS_FLEXIBLE_CONV_EMBED:
            mid_chs = network_def[_BLOCK_EMBED_INDEX][_EMBED_CONV_MID_CHANNELS]
        elif network_def[_BLOCK_EMBED_INDEX][_BLOCK_TYPE] == _TYPE_IS_CONV_EMBED:
            mid_chs = 24 # hand-crafted
        compute = compute + compute_patch_embedding(embed_dim, num_patches, 
            num_chs=self.num_in_channels, patch_size=self.patch_size, 
            return_mac=return_mac, 
            conv_embedding=conv_embedding, 
            mid_chs=mid_chs)
        compute = compute + compute_position_embedding(embed_dim, n_seq=n_seq, return_mac=return_mac)
        
        for i in range(len(network_def)):
            block_def = network_def[i]
            if block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
                compute = compute + compute_transformer_block(block_def, n_seq=n_seq, return_mac=return_mac)
                assert block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED] == block_def[_BLOCK_FFN_IDX][_FFN_EMBED], 'Block {}: embedding dim mismatch'.format(i)
                assert block_def[_BLOCK_ATTN_IDX][_ATTN_EMBED] == embed_dim, 'Block {}: embedding dim is not consistent with patch embedding'.format(i)
            
            elif block_def[_BLOCK_TYPE] == _TYPE_IS_SR_BLOCK:
                
                print(compute)
                # update `img_size`, `num_patches`, `n_seq`, and `embed_dim`
                # We assume that SR blocks have fixed patch size (equal to 2)
                compute = compute + compute_sr_block(img_size, patch_size=self.sr_patch_size, 
                    num_in_channels=block_def[_SR_BLOCK_NUM_IN_CHANNELS], 
                    num_out_channels=block_def[_SR_BLOCK_NUM_OUT_CHANNELS], 
                    distill=self.distill, return_mac=return_mac)
                
                assert block_def[_SR_BLOCK_NUM_IN_CHANNELS] == embed_dim
                assert img_size % self.sr_patch_size == 0
                
                img_size = img_size // self.sr_patch_size
                num_patches = img_size * img_size
                n_seq = 2 if self.distill else 1
                n_seq = n_seq + num_patches
                embed_dim = block_def[_SR_BLOCK_NUM_OUT_CHANNELS]
        
        num_classes = network_def[_BLOCK_HEAD_INDEX][_HEAD_CHANNEL]
        
        # assume using both classification and ditillation heads
        head_compute = compute_head(embed_dim, num_classes=num_classes, n_seq=n_seq, return_mac=return_mac)
        if self.distill:
            head_compute = head_compute * 2
        compute = compute + head_compute
        
        return compute


if __name__ == '__main__':
    
    from functools import partial
    
    
    compute_mac = ComputationEstimator(distill=True, input_resolution=224, patch_size=16)
    
    vit_t_network_def = ((0, 192), 
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (1, (192, 3, 64), (192, 768), 1),
                   (2, 192, 1000))
    
    mac = get_compute_from_network_def(network_def=vit_t_network_def, return_mac=True)
    mac_obj = compute_mac(network_def=vit_t_network_def)
    assert mac == mac_obj
    print("MAC: {0:.4E}".format(mac_obj))
    
    
    vit_s_network_def = ((0, 384), 
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (1, (384, 6, 64), (384, 1536), 1),
                   (2, 384, 1000)
                   )
    
    mac = get_compute_from_network_def(network_def=vit_s_network_def, return_mac=True)
    mac_obj = compute_mac(network_def=vit_s_network_def)
    assert mac == mac_obj
    print("MAC: {0:.2E}".format(mac))
    
    vit_b_network_def = ((0, 768), 
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (1, (768, 12, 64), (768, 3072), 1),
                       (2, 768, 1000))
    
    vit_b_network_def = ((0, 384), 
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (1, (384, 8, 64), (384, 1536), 1),
        (2, 384, 1000))
    
    mac = get_compute_from_network_def(network_def=vit_b_network_def, return_mac=True)
    mac_obj = compute_mac(network_def=vit_b_network_def)
    assert mac == mac_obj
    print("MAC: {0:.2E}".format(mac))
    
    network_def = ((0, 192), (1, (192, 3, 64), (192, 480), 1), (1, (192, 4, 64), (192, 480), 1), (1, (192, 3, 64), (192, 480), 1), (1, (192, 3, 64), (192, 480), 1), (1, (192, 3, 64), (192, 480), 1), (1, (192, 3, 64), (192, 480), 1), (1, (192, 3, 64), (192, 640), 1), (1, (192, 3, 64), (192, 480), 1), (1, (192, 3, 64), (192, 480), 1), (1, (192, 4, 64), (192, 480), 1), (1, (192, 3, 64), (192, 480), 1), (1, (192, 3, 64), (192, 480), 0), (1, (192, 3, 64), (192, 480), 1), (1, (192, 3, 64), (192, 480), 0), (1, (192, 4, 64), (192, 800), 1), (1, (192, 4, 64), (192, 480), 1), (2, 192, 1000))
            
    mac = get_compute_from_network_def(network_def=network_def, return_mac=True)
    print("MAC: {0:.2E}".format(mac))
    
    compute_mac_192 = ComputationEstimator(distill=True, input_resolution=192, patch_size=16)
    network_def_192 = ((0, 224), (1, (224, 3, 64), (224, 480), 1), (1, (224, 3, 64), (224, 640), 1), (1, (224, 3, 64), (224, 640), 1), (1, (224, 3, 64), (224, 640), 1), (1, (224, 3, 64), (224, 640), 1), (1, (224, 3, 64), (224, 480), 1), (1, (224, 4, 64), (224, 640), 1), (1, (224, 3, 64), (224, 800), 1), (1, (224, 3, 64), (224, 480), 1), (1, (224, 4, 64), (224, 800), 1), (1, (224, 4, 64), (224, 480), 1), (1, (224, 3, 64), (224, 640), 1), (1, (224, 6, 64), (224, 480), 1), (1, (224, 5, 64), (224, 800), 1), (1, (224, 4, 64), (224, 960), 1), (2, 224, 1000))
    get_mac_192 = partial(get_compute_from_network_def, resolution=192)
    print(get_mac_192(network_def_192))
    
    assert get_mac_192(network_def_192) == compute_mac_192(network_def_192)
    
    # For ViT-SR
    compute_mac_r224_p14 = ComputationEstimator(distill=False, input_resolution=224, patch_size=14)
    network_def = ((4, 192), 
        (1, (192, 3, 64), (192, 768), 1),
        (1, (192, 3, 64), (192, 768), 1),
        (1, (192, 3, 64), (192, 768), 1),
        (1, (192, 3, 64), (192, 768), 1),
        (3, 192, 384), 
        (1, (384, 6, 64), (384, 1536), 1),
        (1, (384, 6, 64), (384, 1536), 1),
        (1, (384, 6, 64), (384, 1536), 1),
        (1, (384, 6, 64), (384, 1536), 1),
        (3, 384, 768), 
        (1, (768, 12, 64), (768, 3072), 1),
        (1, (768, 12, 64), (768, 3072), 1),
        (1, (768, 12, 64), (768, 3072), 1),
        (1, (768, 12, 64), (768, 3072), 1),
        (2, 768, 1000))
    mac = compute_mac_r224_p14(network_def)
    print('ViT-SR-Tiny MAC: {0:.4E}'.format(mac))
    
    network_def = ((5, 320, 32), 
        (1, (320, 6, 64), (320, 960), 1),
        (1, (320, 6, 64), (320, 960), 1),
        (1, (320, 6, 64), (320, 960), 1),
        (1, (320, 6, 64), (320, 960), 1),
        (1, (320, 6, 64), (320, 960), 1),
        (1, (320, 6, 64), (320, 960), 1),
        (3, 320, 640), 
        (1, (640, 12, 64), (640, 1920), 1),
        (1, (640, 12, 64), (640, 1920), 1),
        (1, (640, 12, 64), (640, 1920), 1),
        (1, (640, 12, 64), (640, 1920), 1),
        (1, (640, 12, 64), (640, 1920), 1),
        (1, (640, 12, 64), (640, 1920), 1),
        (3, 640, 1280), 
        (1, (1280, 12, 64), (1280, 3840), 1),
        (1, (1280, 12, 64), (1280, 3840), 1),
        (1, (1280, 12, 64), (1280, 3840), 1),
        (1, (1280, 12, 64), (1280, 3840), 1),
        (1, (1280, 12, 64), (1280, 3840), 1),
        (1, (1280, 12, 64), (1280, 3840), 1),
        (2, 1280, 1000))
    mac = compute_mac_r224_p14(network_def)
    print('ViT-ResNAS-Small-Largest MAC: {0:.4E}'.format(mac))
    
    #network_def = ((4, 240), (1, (240, 7, 32), (240, 960), 1), (1, (240, 6, 32), (240, 960), 1), (1, (240, 7, 32), (240, 800), 1), (1, (240, 8, 32), (240, 960), 1), (1, (240, 7, 32), (240, 880), 1), (1, (240, 8, 32), (240, 880), 1), (1, (240, 6, 32), (240, 800), 1), (3, 240, 640), (1, (640, 10, 48), (640, 1120), 1), (1, (640, 14, 48), (640, 1760), 1), (1, (640, 14, 48), (640, 1920), 1), (1, (640, 16, 48), (640, 1760), 1), (1, (640, 14, 48), (640, 1440), 1), (1, (640, 16, 48), (640, 1760), 1), (1, (640, 16, 48), (640, 1920), 1), (3, 640, 880), (1, (880, 16, 64), (880, 3200), 1), (1, (880, 10, 64), (880, 3840), 1), (1, (880, 16, 64), (880, 3840), 1), (1, (880, 12, 64), (880, 3840), 0), (1, (880, 12, 64), (880, 3200), 1), (1, (880, 16, 64), (880, 3520), 1), (1, (880, 14, 64), (880, 3520), 1), (2, 880, 1000))
    #mac = compute_mac_r224_p14(network_def)
    #print('Teste MAC: {0:.4E}'.format(mac))
    
    