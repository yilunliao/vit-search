'''
    This file defines functions used to finetune models at higher resolution.
'''


import torch
import math


def load_interpolated_state_dict(model_state_dict, ckpt_path):
    '''
        Load and interpolate state dict. 
        Return the interpolated state dict for finetuning at higher resolution
    '''
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model_ema' in state_dict:
        state_dict = state_dict['model_ema']
    else:
        state_dict = state_dict['model']
    state_dict = state_dict_interpolate_pos_embed(model_state_dict, state_dict)
    return state_dict


def state_dict_interpolate_pos_embed(model_state_dict, state_dict):
    
    # check whether there is distillation token
    assert 'tokens' in model_state_dict.keys()
    token_shape = model_state_dict['tokens'].shape
    num_tokens = 2 if token_shape[1] == 2 else 1
    
    # interpolate pos_embed
    for k in model_state_dict.keys():
        assert k in state_dict.keys()
        
        if 'pos_embed' in k:
            if 'blocks' in k:
                # spatial reduction
                token_pos_embed = None
                patch_pos_embed = state_dict[k]
                new_size = int(math.sqrt((model_state_dict[k].shape[1])))
            else:
                # first stage (remove cls and dst tokens)
                token_pos_embed = state_dict[k][:, 0:num_tokens, :]
                patch_pos_embed = state_dict[k][:, num_tokens::, :]
                new_size = int(math.sqrt((model_state_dict[k].shape[1] - num_tokens)))
            
            orig_size = int(math.sqrt(patch_pos_embed.shape[1]))
            
            if new_size == orig_size:
                continue
                
            embed_size = patch_pos_embed.shape[2]
            
            patch_pos_embed = patch_pos_embed.reshape(1, orig_size, orig_size, embed_size).permute(0, 3, 1, 2)
            patch_pos_embed = torch.nn.functional.interpolate(patch_pos_embed, 
                size=(new_size, new_size), mode='bicubic', align_corners=False)
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            
            if token_pos_embed is None:
                pos_embed = patch_pos_embed
            else:
                pos_embed = torch.cat((token_pos_embed, patch_pos_embed), dim=1)
            
            state_dict[k] = pos_embed
    
    return state_dict