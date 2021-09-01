'''
    This file defines functions used to generate network_def for evolutionary search.
    
    Network def looks like:
    ((0, 256), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (1, (256, 4, 64), (256, 1024), 1), 
    (2, 256, 1000))
'''
import numpy as np
import copy
import time

from .compute_flop_mac import get_compute_from_network_def, ComputationEstimator


# enum for network_def
_BLOCK_EMBED_INDEX = 0
_EMBED_CHANNEL = 1
_BLOCK_HEAD_INDEX = -1
_HEAD_CHANNEL  = 2

_BLOCK_TYPE = 0
_TYPE_IS_EMBED = 0
_TYPE_IS_TRANS = 1 # transformer block
_TYPE_IS_HEAD = 2
_TYPE_IS_SR = 3
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

_EVOLVE_NETWORK_DEF_RESORUCE_LOWER_BOUND = 0.975


def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def tupleit(t):
    return tuple(map(tupleit, t)) if isinstance(t, (tuple, list)) else t


def update_embed_size(network_def):
    '''
        Update `embed_size` after SR blocks.
    '''
    embed_size = network_def[_BLOCK_EMBED_INDEX][_EMBED_CHANNEL]
    for i in range(1, len(network_def)):
        if network_def[i][_BLOCK_TYPE] == _TYPE_IS_TRANS:
            network_def[i][_BLOCK_ATTN_IDX][_ATTN_EMBED] = embed_size
            network_def[i][_BLOCK_FFN_IDX][_FFN_EMBED] = embed_size
        elif network_def[i][_BLOCK_TYPE] == _TYPE_IS_HEAD:
            network_def[i][_EMBED_CHANNEL] = embed_size
        elif network_def[i][_BLOCK_TYPE] == _TYPE_IS_SR:
            network_def[i][_SR_IN_CHANNEL] = embed_size
            embed_size = network_def[i][_SR_OUT_CHANNEL] 
        else:
            raise ValueError()
    return network_def


def update_depth(network_def, num_channels_to_keep):
    '''
        A block will be removed if its previous block is removed and this block
        has an option to be removed.
        
        For example, 
        block:      1 -> 2 -> 3 -> 4
        removable:  X    X    Y    Y
        current:    1    1    0    1
        updated:    1    1    0   1->0
    '''
    remove_block = False
    for i in range(len(network_def)):
        block_def = network_def[i]
        num_channels_to_keep_block = num_channels_to_keep[i]
        if block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
            if num_channels_to_keep_block['layer'] is None:
                remove_block = False
            else:
                if remove_block:
                    network_def[i][_BLOCK_EXISTS_IDX] = 0
                    continue
                else:
                    if not block_def[_BLOCK_EXISTS_IDX]:
                        remove_block = True
    return network_def     


def prune_random_one(network_def, num_channels_to_keep, prune_embed=True, prune_block=True):
    
    network_def = copy.deepcopy(network_def)
    num_blocks = len(network_def) - 1 # not simplify head
    start_idx = 0 if prune_embed else 1
    block_idx = np.random.randint(start_idx, num_blocks)
    
    if not prune_embed:
        block_type = network_def[block_idx][_BLOCK_TYPE]
        while block_type in [_TYPE_IS_EMBED, _TYPE_IS_SR, _TYPE_IS_CONV_EMBED, _TYPE_IS_FLEXIBLE_CONV_EMBED]:
            block_idx = np.random.randint(start_idx, num_blocks)
            block_type = network_def[block_idx][_BLOCK_TYPE]
    #(print)('Block idx:', block_idx)
    
    block_def = network_def[block_idx]
    num_channels_to_keep_block = num_channels_to_keep[block_idx]
    if block_def[_BLOCK_TYPE] in [_TYPE_IS_EMBED, _TYPE_IS_CONV_EMBED, _TYPE_IS_FLEXIBLE_CONV_EMBED]:
        for i in num_channels_to_keep_block:
            if i < block_def[_EMBED_CHANNEL]:
                network_def[block_idx][_EMBED_CHANNEL] = int(i)
                break
        network_def = update_embed_size(network_def)
        
    elif block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
        if num_channels_to_keep_block['layer'] is not None and prune_block:
            simplify_option = 3
        else:
            simplify_option = 2
        simplify_choice = np.random.randint(simplify_option)
        #print('Simplify choice:', simplify_choice)
        
        if simplify_choice == 0: # attn
            attn_head_choices = num_channels_to_keep_block['attn']
            for i in attn_head_choices:
                i = int(i)
                i = i // (block_def[_BLOCK_ATTN_IDX][_ATTN_HEAD_DIM])
                if i < block_def[_BLOCK_ATTN_IDX][_ATTN_NUM_HEAD]:
                    network_def[block_idx][_BLOCK_ATTN_IDX][_ATTN_NUM_HEAD] = i
                    break
        elif simplify_choice == 1: # ffn (mlp)
            ffn_choices = num_channels_to_keep_block['mlp']
            for i in ffn_choices:
                i = int(i)
                if i < block_def[_BLOCK_FFN_IDX][_FFN_HIDDEN]:
                    network_def[block_idx][_BLOCK_FFN_IDX][_FFN_HIDDEN] = i
                    break
        elif simplify_choice == 2: # layer
            layer_choices = num_channels_to_keep_block['layer']
            keep_layer = np.random.choice(layer_choices)
            keep_layer = int(keep_layer)
            if not keep_layer:
                network_def[block_idx][_BLOCK_EXISTS_IDX] = 0
                network_def = update_depth(network_def, num_channels_to_keep)
                
    elif block_def[_BLOCK_TYPE] == _TYPE_IS_SR:
        # SR block
        for i in num_channels_to_keep_block:
            if i < block_def[_SR_OUT_CHANNEL]:
                network_def[block_idx][_SR_OUT_CHANNEL] = int(i)
                network_def = update_embed_size(network_def)
                break
    
    else:
        raise ValueError()
        
    return network_def


def reduce_constraint(network_def, num_channels_to_keep, constraint, compute_resource):
    '''
        Randomly prune layers or blocks of `largest_network_def` 
        based on `num_channels_to_keep` until 
        resources estimated by `compute_resource` is lower than `constraint`.
        
        At first, prune # of heads and FFN.
        If resource constraint cannot be met, prune both embedding dimension 
        and blocks.
    '''
    _simplify_all_threshold = 100 # prevent not meeting constraint
    
    if isinstance(network_def, tuple):
        network_def = listit(network_def)    
    simplify_times = 0
    while compute_resource(network_def) > constraint:
        if simplify_times < _simplify_all_threshold:
            prune_embed = False
            prune_block = False
        else:
            prune_embed = True
            prune_block = True
        network_def = prune_random_one(network_def, num_channels_to_keep, 
            prune_embed=prune_embed, prune_block=prune_block)
        simplify_times = simplify_times + 1
    return network_def


def random_sample_embed_depth(largest_network_def, num_channels_to_keep):
    network_def = copy.deepcopy(largest_network_def)
    if isinstance(network_def, tuple):
        network_def = listit(network_def)
        
    for i in range(len(network_def)):
        block_def = network_def[i]
        num_channels_to_keep_block = num_channels_to_keep[i]
        if block_def[_BLOCK_TYPE] in [_TYPE_IS_EMBED, _TYPE_IS_CONV_EMBED, _TYPE_IS_FLEXIBLE_CONV_EMBED]:
            embed_size = np.random.choice(num_channels_to_keep_block) #block_def[_EMBED_CHANNEL]
            network_def[i][_EMBED_CHANNEL] = int(embed_size)
            network_def = update_embed_size(network_def)
        elif block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
            if num_channels_to_keep_block['layer'] is not None:
                keep_layer = np.random.choice(num_channels_to_keep_block['layer'])
                keep_layer = int(keep_layer)
                if not keep_layer:
                    network_def[i][_BLOCK_EXISTS_IDX] = 0
        elif block_def[_BLOCK_TYPE] == _TYPE_IS_SR:
            embed_size = np.random.choice(num_channels_to_keep_block) 
            network_def[i][_SR_OUT_CHANNEL] = int(embed_size)
            network_def = update_embed_size(network_def)
    #print(network_def)
    network_def = update_depth(network_def, num_channels_to_keep)
    return network_def

    
def gen_random_func(largest_network_def, num_channels_to_keep, constraint, compute_resource):
    network_def = random_sample_embed_depth(largest_network_def, num_channels_to_keep)
    while compute_resource(network_def) < _EVOLVE_NETWORK_DEF_RESORUCE_LOWER_BOUND * constraint:
        network_def = random_sample_embed_depth(largest_network_def, num_channels_to_keep)
    
    network_def = reduce_constraint(network_def, num_channels_to_keep, constraint, compute_resource)
    if isinstance(network_def, list):
        network_def = tupleit(network_def)
    return network_def   


def gen_random_network_def(largest_network_def, num_channels_to_keep, constraint, compute_resource): 
    r_network_def = gen_random_func(largest_network_def, num_channels_to_keep, constraint, compute_resource)
    r_resource = compute_resource(r_network_def)
    
    while r_resource > constraint or r_resource < _EVOLVE_NETWORK_DEF_RESORUCE_LOWER_BOUND * constraint:
        r_network_def = gen_random_func(largest_network_def, num_channels_to_keep, constraint, compute_resource)
        r_resource = compute_resource(r_network_def)
    return r_network_def    
    

def mutate_func(parent_network_def, num_channels_to_keep, m_prob):
    network_def = copy.deepcopy(parent_network_def)
    if isinstance(network_def, tuple):
        network_def = listit(network_def)
    for i in range(len(network_def)):
        block_def = network_def[i]
        num_channels_to_keep_block = num_channels_to_keep[i]
        if block_def[_BLOCK_TYPE] in [_TYPE_IS_EMBED, _TYPE_IS_CONV_EMBED, _TYPE_IS_FLEXIBLE_CONV_EMBED]:
            if np.random.uniform() <= m_prob:
                embed_size = int(np.random.choice(num_channels_to_keep_block))
                network_def[i][_EMBED_CHANNEL] = embed_size
                network_def = update_embed_size(network_def)
                
        elif block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
            # attn
            if np.random.uniform() <= m_prob:
                attn_head_choice = int(np.random.choice(num_channels_to_keep_block['attn']))
                attn_head_choice = attn_head_choice // block_def[_BLOCK_ATTN_IDX][_ATTN_HEAD_DIM]
                network_def[i][_BLOCK_ATTN_IDX][_ATTN_NUM_HEAD] = attn_head_choice
            # ffn (mlp)
            if np.random.uniform() <= m_prob:
                ffn_choice = int(np.random.choice(num_channels_to_keep_block['mlp']))
                network_def[i][_BLOCK_FFN_IDX][_FFN_HIDDEN] = ffn_choice
            # block
            if num_channels_to_keep_block['layer'] is None:
                continue
            if np.random.uniform() <= m_prob:
                #keep_layer = int(np.random.choice(num_channels_to_keep_block['layer']))
                #if not keep_layer:
                #    network_def[i][_BLOCK_EXISTS_IDX] = 0
                #else:
                #    network_def[i][_BLOCK_EXISTS_IDX] = 1
                if block_def[_BLOCK_EXISTS_IDX]:
                    network_def[i][_BLOCK_EXISTS_IDX] = 0
                else:
                    network_def[i][_BLOCK_EXISTS_IDX] = 1
                network_def = update_depth(network_def, num_channels_to_keep)
                
        elif block_def[_BLOCK_TYPE] == _TYPE_IS_SR:
            # basically same as `_TYPE_IS_EMBED`
            if np.random.uniform() <= m_prob:
                embed_size = int(np.random.choice(num_channels_to_keep_block))
                network_def[i][_SR_OUT_CHANNEL] = embed_size
                network_def = update_embed_size(network_def)
                
        elif block_def[_BLOCK_TYPE] == _TYPE_IS_HEAD:
            pass
        else:
            raise ValueError()
            
    return network_def


def mutate_network_def(parent_network_def, num_channels_to_keep, m_prob, constraint, compute_resource):
    '''
        Call mutate_func() to mutate network_def and check whether 
        the generated network_def has too low resource.
    '''
    _mutate_resource_threshold = _EVOLVE_NETWORK_DEF_RESORUCE_LOWER_BOUND
    m_network_def = mutate_func(parent_network_def, num_channels_to_keep, m_prob)
    m_resource = compute_resource(m_network_def)
    
    while m_resource < _mutate_resource_threshold * constraint or m_resource > constraint:
        m_network_def = mutate_func(parent_network_def, num_channels_to_keep, m_prob)
        m_resource = compute_resource(m_network_def)
    
    if isinstance(m_network_def, list):
        m_network_def = tupleit(m_network_def)
    return m_network_def


def crossover_func(m_network_def, f_network_def, num_channels_to_keep):
    network_def = copy.deepcopy(m_network_def)
    if isinstance(network_def, tuple):
        network_def = listit(network_def)
    
    for i in range(len(network_def)):
        m_block_def = network_def[i]
        #f_block_def = f_network_def[i]
        
        if m_block_def[_BLOCK_TYPE] in [_TYPE_IS_EMBED, _TYPE_IS_CONV_EMBED, _TYPE_IS_FLEXIBLE_CONV_EMBED]:
            if np.random.uniform() <= 0.5:
                embed_size = f_network_def[i][_EMBED_CHANNEL]
                network_def[i][_EMBED_CHANNEL] = embed_size
                network_def = update_embed_size(network_def)
                
        elif m_block_def[_BLOCK_TYPE] == _TYPE_IS_TRANS:
            # attn
            if np.random.uniform() <= 0.5:
                attn_head_choice = f_network_def[i][_BLOCK_ATTN_IDX][_ATTN_NUM_HEAD]
                network_def[i][_BLOCK_ATTN_IDX][_ATTN_NUM_HEAD] = attn_head_choice
            # ffn (mlp)
            if np.random.uniform() <= 0.5:
                ffn_choice = f_network_def[i][_BLOCK_FFN_IDX][_FFN_HIDDEN]
                network_def[i][_BLOCK_FFN_IDX][_FFN_HIDDEN] = ffn_choice
            # block
            if np.random.uniform() <= 0.5:
                network_def[i][_BLOCK_EXISTS_IDX] = f_network_def[i][_BLOCK_EXISTS_IDX]
                network_def = update_depth(network_def, num_channels_to_keep)
                
        elif m_block_def[_BLOCK_TYPE] == _TYPE_IS_SR:
            if np.random.uniform() <= 0.5:
                network_def[i][_SR_OUT_CHANNEL] = f_network_def[i][_SR_OUT_CHANNEL]
                network_def = update_embed_size(network_def)
        
        elif m_block_def[_BLOCK_TYPE] == _TYPE_IS_HEAD:
            pass
        else:
            raise ValueError()
                
    return network_def


def crossover_network_def(m_network_def, f_network_def, num_channels_to_keep, constraint, compute_resource):
    '''
        Call crossover_func() to generate network_def and check whether 
        the generated network_def has too low resource.
    '''
    _crossover_resource_threshold = _EVOLVE_NETWORK_DEF_RESORUCE_LOWER_BOUND
    c_network_def = crossover_func(m_network_def, f_network_def, num_channels_to_keep)
    c_resource = compute_resource(c_network_def)
    
    while c_resource < _crossover_resource_threshold * constraint or c_resource > constraint:
        c_network_def = crossover_func(m_network_def, f_network_def, num_channels_to_keep)
        c_resource = compute_resource(c_network_def)
    
    if isinstance(c_network_def, list):
        c_network_def = tupleit(c_network_def)
    return c_network_def


if __name__ == '__main__':
    
    num_channels_to_keep = []
    # stage 1
    embed = np.array([256, 224, 192, 176, 160])
    block = {'attn': np.array([256, 192, 128]), 'mlp': np.array([768, 640, 512, 384]), 'layer': None}  
    block_skip = copy.deepcopy(block)
    block_skip['layer'] = np.array([256, 256, 256, 0])
    
    num_channels_to_keep.append(embed)
    num_channels_to_keep.append(block)
    for i in range(3):
        num_channels_to_keep.append(block_skip)
        num_channels_to_keep.append(block)
    
    # stage 2
    embed = np.array([512, 448, 384, 352, 320])
    block = {'attn': np.array([512, 384, 256]), 'mlp': np.array([1536, 1280, 1024, 768]), 'layer': None}  
    block_skip = copy.deepcopy(block)
    block_skip['layer'] = np.array([512, 512, 512, 0])
    
    num_channels_to_keep.append(embed)
    num_channels_to_keep.append(block)
    for i in range(3):
        num_channels_to_keep.append(block_skip)
        num_channels_to_keep.append(block)
        
    # stage 3
    embed = np.array([1024, 896, 768, 704, 640])
    block = {'attn': np.array([768, 640, 512]), 'mlp': np.array([3072, 2560, 2048, 1536]), 'layer': None}  
    block_skip = copy.deepcopy(block)
    block_skip['layer'] = np.array([1024, 1024, 1024, 0])
    
    num_channels_to_keep.append(embed)
    for i in range(4):
        num_channels_to_keep.append(block)
    num_channels_to_keep.append(None)

    
    network_def = ((0, 256), 
        (1, (256, 4, 64), (256, 768), 1),
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
        (1, (512, 8, 64), (512, 1536), 1),
        (3, 512, 1024), 
        (1, (1024, 12, 64), (1024, 3072), 1),
        (1, (1024, 12, 64), (1024, 3072), 1),
        (1, (1024, 12, 64), (1024, 3072), 1),
        (1, (1024, 12, 64), (1024, 3072), 1),
        (2, 1024, 1000))
    
    test_network_def = listit(network_def)
    test_network_def[_BLOCK_EMBED_INDEX][_EMBED_CHANNEL] = 224
    test_network_def[8][_SR_OUT_CHANNEL]  = 320
    test_network_def[16][_SR_OUT_CHANNEL] = 768
    test_network_def = update_embed_size(test_network_def)
    test_network_def = tupleit(test_network_def)
        
    test_network_def = listit(network_def)
    compute_mac = ComputationEstimator(distill=True, input_resolution=224, patch_size=14)
    print('MAC:', compute_mac(test_network_def))
    
    test_network_def = listit(network_def)
    test_network_def = prune_random_one(test_network_def, num_channels_to_keep)
    test_network_def = prune_random_one(test_network_def, num_channels_to_keep)
    test_network_def = prune_random_one(test_network_def, num_channels_to_keep)
    test_network_def = prune_random_one(test_network_def, num_channels_to_keep)
    test_network_def = prune_random_one(test_network_def, num_channels_to_keep)
    test_network_def = prune_random_one(test_network_def, num_channels_to_keep)
    print('MAC:', compute_mac(test_network_def))
    print(test_network_def)
    
    
    start_time = time.time()
    
    test_network_def = gen_random_network_def(largest_network_def=network_def, 
        num_channels_to_keep=num_channels_to_keep, 
        constraint=compute_mac(network_def) * 0.37, 
        compute_resource=compute_mac)
    print('MAC:', compute_mac(test_network_def))
    print(test_network_def)
    

    m_network_def = mutate_network_def(parent_network_def=test_network_def, 
        num_channels_to_keep=num_channels_to_keep, m_prob=0.3, 
        constraint=compute_mac(network_def) * 0.37, 
        compute_resource=compute_mac)
    print('MAC:', compute_mac(m_network_def))
    print(m_network_def)
    
    c_network_def = crossover_network_def(m_network_def=m_network_def, 
        f_network_def=test_network_def, 
        num_channels_to_keep=num_channels_to_keep,
        constraint=compute_mac(network_def) * 0.37, 
        compute_resource=compute_mac)
    print('MAC:', compute_mac(c_network_def))
    print(c_network_def)
    
    print(time.time() - start_time)
    