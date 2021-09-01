from .gen_utils import tupleit, listit, gen_random_network_def, mutate_network_def, crossover_network_def, reduce_constraint
from .compute_flop_mac import get_compute_from_network_def, ComputationEstimator

import numpy as np
import copy
import warnings
import time


_CROSSOVER_SKIP_CHECKING_THRESHOLD = 100 


class Individual():
    def __init__(self, network_def, score=-1):
        self.network_def = network_def
        self.score = score
        
    def __lt__(self, other):
        return self.score < other.score
    
    def __eq__(self, other):
        return self.network_def == other.network_def
    
    def __repr__(self):
        return '(network_def={}, score={})'.format(self.network_def, self.score)
    

# Reference: https://github.com/mit-han-lab/hardware-aware-transformers/blob/c8d6d71903854537d265129bea7c5d162c4ee210/fairseq/evolution.py#L183
class PopulationEvolver():
    def __init__(self, largest_network_def, num_channels_to_keep, constraint, compute_resource):
        self.largest_network_def = largest_network_def
        self.num_channels_to_keep = num_channels_to_keep
        self.constraint = constraint
        self.compute_resource = compute_resource
        
        self.popu = [] # population
        self.history_popu = [] # save all previous networks to prevent evaluating the same networks
        
    
    def random_sample(self, num_samples):
        popu_idx = 0
        while popu_idx < num_samples:
            network_def = gen_random_network_def(largest_network_def=self.largest_network_def, 
                num_channels_to_keep=self.num_channels_to_keep, 
                constraint=self.constraint, 
                compute_resource=self.compute_resource)
            new_ind = Individual(network_def)
            if new_ind not in self.popu and new_ind not in self.history_popu:
                self.popu.append(new_ind)
                popu_idx = popu_idx + 1
        return
    
    
    def update_history(self):
        for i in range(len(self.popu)):
            if self.popu[i] not in self.history_popu:
                self.history_popu.append(self.popu[i])
        #self.history_popu.extend(self.popu)
        self.popu = []
        return
    
    
    def sort_history(self):
        self.history_popu.sort(reverse=True)
        return 
        
    
    def evolve_sample(self, parent_size, mutate_prob, mutate_size, crossover_size=None):
        if self.popu:
            warnings.warn('[evolve_sample] popu is not empty.')
        if not self.history_popu:
            warnings.warn('[evolve_sample] history_popu is empty. Use update_history() before evolve_sample().')
            return 
        if parent_size > len(self.history_popu):
            raise ValueError('Parent size is larger than history population size')
        
        self.sort_history()
        if crossover_size is None:
            crossover_size = mutate_size
            
        # mutation
        popu_idx = 0
        while popu_idx < mutate_size:
            parent_idx = np.random.randint(parent_size)
            parent_network_def = self.history_popu[parent_idx].network_def
            network_def = mutate_network_def(parent_network_def, 
                num_channels_to_keep=self.num_channels_to_keep, 
                m_prob=mutate_prob, 
                constraint=self.constraint, 
                compute_resource=self.compute_resource)
            new_ind = Individual(network_def)
            if new_ind not in self.popu and new_ind not in self.history_popu:
                self.popu.append(new_ind)
                popu_idx = popu_idx + 1
            
        # crossover
        popu_idx = 0
        skip_checking_counter = 0 # prevent infinite loop when at later iterations crossover does not produce new samples
        skip_checking_threshold = _CROSSOVER_SKIP_CHECKING_THRESHOLD
        while popu_idx < crossover_size:
            parent_idx = np.random.choice(range(parent_size), size=2, replace=False)
            m_network_def = self.history_popu[parent_idx[0]].network_def
            f_network_def = self.history_popu[parent_idx[1]].network_def
            network_def = crossover_network_def(m_network_def, f_network_def,
                num_channels_to_keep=self.num_channels_to_keep,
                constraint=self.constraint, 
                compute_resource=self.compute_resource)
            new_ind = Individual(network_def)
            if (new_ind not in self.popu and new_ind not in self.history_popu) or skip_checking_counter >= skip_checking_threshold:
                self.popu.append(new_ind)
                popu_idx = popu_idx + 1
                skip_checking_counter = 0
            else: 
                skip_checking_counter = skip_checking_counter + 1
    
        return
    

if __name__ == '__main__':
    
    import math 
    
    '''
    For testing
    '''
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

    largest_network_def = ((0, 256), 
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
    
    def compute_acc(network_def):
        
        def compute_score(network_def):
            score = 0
            embed_size = 0
            for i in range(len(network_def)):
                
                depth_factor = (11.0 + i) / 8.0
                
                if i == 0:
                    embed_size = network_def[i][1]
                    continue
                if network_def[i][0] == 1:
                    #block_score = network_def[i][1][1] * network_def[i][1][2] * depth_factor * 2
                    #block_score = block_score + network_def[i][2][1] * depth_factor
                    #block_score = block_score * math.sqrt(network_def[i][1][0])
                    #score = score + block_score
                    embed_size = network_def[i][1][0]
                    score = score + network_def[i][1][1] * network_def[i][1][2] * depth_factor * 2 * math.sqrt(embed_size)
                    score = score + network_def[i][2][1] * depth_factor * math.sqrt(embed_size)
                    
            #score = score * math.sqrt(embed_size)
            return score
        
        max_score = compute_score(largest_network_def)
        score = compute_score(network_def)
        return score / max_score
    
    compute_mac_r224_p14 = ComputationEstimator(distill=True, input_resolution=224, patch_size=14)
    popu_evolve = PopulationEvolver(largest_network_def=largest_network_def, 
                                    num_channels_to_keep=num_channels_to_keep, 
                                    constraint=compute_mac_r224_p14(largest_network_def)*0.37, 
                                    compute_resource=compute_mac_r224_p14)
    np.random.seed(0)
    search_start_time = time.time()
    for search_iter in range(30):
        
        iter_start_time = time.time()
        if search_iter == 0:
            popu_evolve.random_sample(num_samples=500)
        else:
            popu_evolve.evolve_sample(parent_size=75, mutate_prob=0.3, mutate_size=75)
        
        for idx, ind in enumerate(popu_evolve.popu):
            ind.score = compute_acc(ind.network_def)
            
        popu_evolve.update_history()
        popu_evolve.sort_history()
        
        print('Iter: {} - Max acc. = {}, Time = {}'.format(search_iter, 
            popu_evolve.history_popu[0].score, time.time() - iter_start_time))
    print('Search time: {}'.format(time.time() - search_start_time))