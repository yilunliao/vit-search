import torch
import torch.nn as nn
import numpy as np
import math


_NUM_WARMUP_EPOCHS_CHANNEL    = 5   # For ChannelDrop

'''
    Channel Drop can be used to drop # of heads, FFN hidden channels, embedding dimension, 
    and an entire layer.
    
    When dropping an entire layer, use `num_channels_to_keep` = [embedding_dim, embedding_dim, 0]
    to set the probability of using different number of layers.
'''
class ChannelDrop(nn.Module):
    def __init__(self, num_channels_to_keep=None, 
                 num_warmup_epochs=_NUM_WARMUP_EPOCHS_CHANNEL,
                 example_per_arch=None, single_arch=False):
        '''
            `example_per_arch`: For each batch, each width would have `example_per_arch` examples 
                to train.
            `single_arch`: Disable multi-architecture training.
            
            Assume the operation is performed on GPU.
        '''
        super(ChannelDrop, self).__init__()
        
        assert num_channels_to_keep is not None
        assert example_per_arch is not None
        assert isinstance(num_channels_to_keep, np.ndarray), 'num_channels_to_keep data type error'
        
        self.num_channels_to_keep = np.sort(num_channels_to_keep)[::-1]
        
        self.epoch_now = None
        self.num_warmup_epochs = num_warmup_epochs

        self.example_per_arch = example_per_arch
        
        # Use single arch for each forward pass or not
        self.single_arch = single_arch
        
        self.mask = None            # for training
        self.mask_all_true = None   # for testing
        
        # number of layer channels considered in the current epoch
        self.num_layer_config = None
        
        # for debug purpose
        '''
            Use `set_random_fixed_mask` or `set_fixed_mask`
        '''
        self.fixed_mask = None
        
        
    def forward(self, x):
        '''
            Input: 
                x: (B, N, C)
            
            Output:
                x: (B, N, C) (masked)
                masked: (B, 1, C) (torch.bool Tensor)
                
            `self.fixed_mask` is for debugging.
            If `self.fixed_mask` is not None, 
            the mask will always be used.
        '''
        # for debug
        if self.fixed_mask is not None:
            assert self.fixed_mask.shape[0] == 1
            B = x.shape[0]
            mask = self.fixed_mask.repeat(B, 1, 1)
            x = x * mask
            return x, mask
        
        if self.training:
            if self.mask is None:
                self.set_mask(x)
                
            mask = self.forward_mask(x)
            x = x * mask
            
        else:
            if self.mask_all_true is None or self.mask_all_true.shape[0] != x.shape[0]:
                B, N, C = x.shape
                self.mask_all_true = torch.ones((B, 1, C), dtype=torch.bool).cuda()
            mask = self.mask_all_true

        return x, mask
    
    
    def forward_mask(self, x):
        permutation_indices = torch.randperm(self.mask.shape[0])
        mask = self.mask[permutation_indices, ...]
        
        if not self.single_arch:
            assert x.shape[0] % self.example_per_arch == 0, 'In forward(), batch size is not divisible by sub-batch size (examples per arch).'
            num_valid_mask = x.shape[0] // self.example_per_arch
         
            if mask.shape[0] != num_valid_mask:
                mask = mask[0:num_valid_mask, :, :]
                
            #mask = torch.repeat_interleave(mask, self.example_per_arch, dim=0)
            mask = mask.repeat(self.example_per_arch, 1, 1)
            
        else:
            mask = mask[0:1, ...]   
            mask = mask.repeat(x.shape[0], 1, 1) #torch.repeat(mask, x.shape[0], dim=0)
            
        return mask
    

    def set_mask(self, inputs):
        '''
            Generate mask, which will be permuted and duplicated during forward()
            
            Duplication is to make sure that each architecture has `self.example_per_arch` examples
            for each batch of data.
        '''
        
        B, N, C = inputs.shape
        assert B % self.example_per_arch == 0, 'Batch size is not divisible by sub-batch size (examples per arch).'
        assert (self.num_channels_to_keep <= C).all(), 'Some elements in num_channels_to_keep is larger than channel size.'
        assert max(self.num_channels_to_keep) == C, 'Maximum channel not in num_channels_to_keep'
        
        num_channels_to_keep = self.num_channels_to_keep
        assert B >= len(
            num_channels_to_keep), 'The batch size is smaller than the number of channels to keep.'
        
        if self.num_warmup_epochs == 0:
            num_layer_config = len(num_channels_to_keep)
        else:
            #num_layer_config = min(
            #    math.ceil((self.epoch_now + 1) / self.num_warmup_epochs * len(num_channels_to_keep)),
            #    len(num_channels_to_keep))
            num_layer_config = min(
                1 + math.floor(self.epoch_now * (len(num_channels_to_keep) - 1) / self.num_warmup_epochs), 
                len(num_channels_to_keep)
                )
            num_layer_config = max(num_layer_config, 1)
            
        self.num_layer_config = num_layer_config
        
        counter = 0
        num_cycles = math.ceil((B // self.example_per_arch) / self.num_layer_config)
        
        if self.single_arch:
            num_cycles = 1
        
        self.mask = torch.zeros((self.num_layer_config * num_cycles, 1, C), dtype=torch.bool).cuda()
        
        for image_index in range(self.mask.shape[0]):
            self.mask[image_index, :, :num_channels_to_keep[counter]] = True
            counter += 1
            if counter == self.num_layer_config:
                counter = 0


    def set_epoch(self, epoch_now):
        self.epoch_now = epoch_now
        self.reset_mask()

    
    def extra_repr(self):
        output_str = 'num_channels_to_keep={}, num_warmup_epochs={}, example_per_arch={}'.format(self.num_channels_to_keep, 
            self.num_warmup_epochs, self.example_per_arch)
        if self.single_arch:
            output_str += ', single_arch={}'.format(self.single_arch)
        return output_str
    
    
    def set_random_fixed_mask(self):
        #assert self.mask is not None
        
        if self.mask is None:
            # generate random tensor to get mask
            random_inputs = torch.randn((self.example_per_arch * len(self.num_channels_to_keep), 1, max(self.num_channels_to_keep)))
            self.set_mask(random_inputs)
        
        permutation_indices = torch.randperm(self.mask.shape[0])
        mask = self.mask[permutation_indices, ...]
        mask = mask[0:1, ...]
        self.set_fixed_mask(mask)
        
    
    def set_fixed_mask(self, mask):
        assert len(mask.shape) == 3
        assert mask.shape[0] == 1
        self.fixed_mask = mask
        
    
    def reset_mask(self):
        self.mask = None
        self.mask_all_true = None
        self.fixed_mask = None
        self.num_layer_config = None
         
    
if __name__ == '__main__':
    
    import time
    temp = torch.ones((8, 1, 1920), dtype=torch.bool)
    start = time.time()
    for i in range(100):
        temp_repeat = torch.repeat_interleave(temp, 16, dim=0)
    print(time.time() - start)
    
    start = time.time()
    for i in range(100):
        temp_cat = temp.repeat(16, 1, 1)
    print(time.time() - start)
    