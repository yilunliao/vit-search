'''
    Add some functions like pruning attention k, q, v
'''


import torch
import copy


def get_qkv_subnet_state_dict(qkv_source, qkv_subnet):
    
    # nn.Linear.weight shape: out_features, in_features
    
    sub_shape = qkv_subnet.shape
    assert sub_shape[0] % 3 == 0
    sub_head_num_dim = sub_shape[0] // 3
    
    src_shape = qkv_source.shape
    assert src_shape[0] % 3 == 0
    src_head_num_dim = src_shape[0] // 3
    
    q_sub = qkv_source[0:sub_head_num_dim, ...]
    k_sub = qkv_source[src_head_num_dim:src_head_num_dim+sub_head_num_dim, ...]
    v_sub = qkv_source[2*src_head_num_dim:2*src_head_num_dim+sub_head_num_dim, ...]
    
    qkv_sub = torch.cat([q_sub, k_sub, v_sub], dim=0)
    
    if len(sub_shape) == 2:
        qkv_sub = qkv_sub[:, 0:sub_shape[1]]
    
    return qkv_sub
    

def get_sub_state_dict(source_dict, sub_dict):
    
    trg_dict = copy.deepcopy(sub_dict)
    
    for k in sub_dict.keys():
        if 'qkv' in k:
            trg_dict[k] = get_qkv_subnet_state_dict(source_dict[k], sub_dict[k])
        
        else: # proj
            if len(sub_dict[k].shape) == 1:
                trg_dict[k] = source_dict[k][0:sub_dict[k].shape[0]]
            elif len(sub_dict[k].shape) == 2:
                trg_dict[k] = source_dict[k][0:sub_dict[k].shape[0], 0:sub_dict[k].shape[1]] 
            elif len(sub_dict[k].shape) == 3:
                trg_dict[k] = source_dict[k][0:sub_dict[k].shape[0], 0:sub_dict[k].shape[1], 0:sub_dict[k].shape[2]]
            elif len(sub_dict[k].shape) == 4:
                assert sub_dict[k].shape[2] == source_dict[k].shape[2]
                assert sub_dict[k].shape[3] == source_dict[k].shape[3]
                trg_dict[k] = source_dict[k][0:sub_dict[k].shape[0], 0:sub_dict[k].shape[1], :, :]
            elif len(sub_dict[k].shape) == 0:
                pass
            else:
                raise ValueError
            
    return trg_dict