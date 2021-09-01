'''
    Patch-Token Mix: https://arxiv.org/pdf/2104.12753.pdf
    
    reference: https://github.com/ChengyueGongR/PatchVisionTransformer/blob/429110382c0afa11b2395b5beb7b7f2934bb4be8/deit/engine.py
'''


import numpy as np
import torch


# From timm
def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, img_index, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    #y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y * lam + y[img_index] * (1. - lam)


def smoothing_one_hot(x, num_classes, smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y = one_hot(x, num_classes, on_value=on_value, off_value=off_value, device=device)
    return y
    

def my_randint(low, high, size=None):
    if low == high:
        high = low + 1
    return np.random.randint(low, high, size=size)
    
    
class SwitchTokenMix():
    '''
        1. Reference: https://github.com/rwightman/pytorch-image-models/blob/9cc7dda6e5fcbbc7ac5ba5d2d44050d2a8e3e38d/timm/data/mixup.py#L54
        2. Use `numpy` for randomness
    '''
    
    def __init__(self, patch_len, switch_prob=0.5, 
        num_classes=1000, smoothing=0.1):
        # num_patches = patch_len ** 2
        self.patch_len = patch_len
        self.switch_prob = switch_prob
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    
    def __repr__(self):
        return '(patch_len={}, switch_prob={})'.format(self.patch_len, self.switch_prob)
    
    
    def _gen_mask(self, patch_len, sel_rate):
        max_length = min(patch_len, int(patch_len * patch_len * sel_rate))
        length = np.random.randint(1, max(1, max_length - 1))
        width = int(patch_len * patch_len * sel_rate) // length
        if width > patch_len:
            width = patch_len
            length = int(patch_len * patch_len * sel_rate) // width
        length_left_ind = np.random.randint(0, max(0, patch_len - length))
        length_right_ind = length_left_ind + length
        width_left_ind = np.random.randint(0, max(0, patch_len - width))
        width_right_ind = width_left_ind + width
    
        mask = torch.zeros(self.patch_len, self.patch_len).cuda()
        mask[length_left_ind:length_right_ind, width_left_ind:width_right_ind] = 1
        return mask
    
    
    def _gen_random_bbox(self):
        '''
            Generate random BBox coordinate
        '''
        lam = np.random.beta(1., 1.)
        max_length = min(self.patch_len, int(self.patch_len * self.patch_len * lam))
        cut_h = my_randint(1, max(1, max_length - 1))
        cut_w = int(self.patch_len * self.patch_len * lam) // cut_h
        if cut_w > self.patch_len:
            cut_w = self.patch_len
            cut_h = int(self.patch_len * self.patch_len * lam) // cut_w
        
        yl = my_randint(0, max(0, self.patch_len - cut_h), size=2)
        xl = my_randint(0, max(0, self.patch_len - cut_w), size=2)
        
        yl[0] = yl[1]
        xl[0] = xl[1]
        
        yr = yl + cut_h
        xr = xl + cut_w
        
        lam = 1 - (cut_h * cut_w + 0.0) / (self.patch_len * self.patch_len)
        
        return (yl, yr, xl, xr), lam    
            
    
    def _patch_mix_samples(self, samples, bbox, img_index):
        (yl, yr, xl, xr) = bbox
        B, C, H, W = samples.shape
        patch_size = H // self.patch_len
        
        samples[:, :, (patch_size * yl[0]):(patch_size * yr[0]), (patch_size * xl[0]):(patch_size * xr[0])] = \
            samples[img_index][:, :, (patch_size * yl[1]):(patch_size * yr[1]), (patch_size * xl[1]):(patch_size * xr[1])]
        
    
    def _patch_mixup_fn(self, samples, targets):
        img_index = torch.randperm(samples.shape[0])
        
        (yl, yr, xl, xr), lam = self._gen_random_bbox()
        self._patch_mix_samples(samples, bbox=(yl, yr, xl, xr), img_index=img_index)
        
        smoothing_one_hot_targets = smoothing_one_hot(targets, num_classes=self.num_classes, smoothing=self.smoothing)
        B = smoothing_one_hot_targets.shape[0]
        patch_targets = smoothing_one_hot_targets.reshape(B, 1, 1, -1)
        patch_targets = patch_targets.repeat(1, self.patch_len, self.patch_len, 1)
        patch_targets[:, yl[0]:yr[0], xl[0]:xr[0], :] = patch_targets[img_index][:, yl[1]:yr[1], xl[1]:xr[1], :]
        patch_targets = torch.flatten(patch_targets, start_dim=1, end_dim=2)
            
        targets = mixup_target(targets, num_classes=self.num_classes, img_index=img_index, 
            lam=lam, smoothing=self.smoothing)
        
        return samples, targets, patch_targets
    
    
    def _image_mixup_fn(self, samples, targets):
        # Image-level mixup
        # reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py
        img_index = torch.randperm(samples.shape[0])
        lam = np.random.beta(0.8, 0.8) # following DeiT
        samples_flipped = samples[img_index].mul_(1. - lam)
        samples.mul_(lam).add_(samples_flipped)
            
        targets = mixup_target(targets, num_classes=self.num_classes, img_index=img_index, 
            lam=lam, smoothing=self.smoothing)
            
        B = targets.shape[0]
        patch_targets = targets.reshape(B, 1, -1)
        patch_targets = patch_targets.repeat(1, self.patch_len * self.patch_len, 1)
        
        return samples, targets, patch_targets
    
    
    def __call__(self, samples, targets):
        
        #img_index = torch.randperm(samples.shape[0])
        #patch_output_type = None
        
        #samples, targets, patch_targets = self._patch_mixup_fn(samples, targets)
        patch_output_type = 'seq'
        #return samples, targets, patch_targets, patch_output_type
    
        B = samples.shape[0]
        patch_targets = torch.zeros((B, self.patch_len * self.patch_len, self.num_classes), device='cuda')
        new_targets = torch.zeros((B, self.num_classes), device='cuda')
        samples[0:(B//2)], new_targets[0:(B//2)], patch_targets[0:(B//2)] = self._patch_mixup_fn(samples[0:(B//2)], targets[0:(B//2)])
        samples[(B//2)::], new_targets[(B//2)::], patch_targets[(B//2)::] = self._image_mixup_fn(samples[(B//2)::], targets[(B//2)::])
        
        return samples, new_targets, patch_targets, patch_output_type
        '''
        if np.random.rand() < self.switch_prob:
            
            (yl, yr, xl, xr), lam = self._gen_random_bbox()
            
            # Patch-level mixup
            self._patch_mix_samples(samples, bbox=(yl, yr, xl, xr), img_index=img_index)
            
            # token-level label
            smoothing_one_hot_targets = smoothing_one_hot(targets, num_classes=self.num_classes, smoothing=self.smoothing)
            B = smoothing_one_hot_targets.shape[0]
            patch_targets = smoothing_one_hot_targets.reshape(B, 1, 1, -1)
            patch_targets = patch_targets.repeat(1, self.patch_len, self.patch_len, 1)
            patch_targets[:, yl[0]:yr[0], xl[0]:xr[0], :] = patch_targets[img_index][:, yl[1]:yr[1], xl[1]:xr[1], :]
            patch_targets = torch.flatten(patch_targets, start_dim=1, end_dim=2)
            
            targets = mixup_target(targets, num_classes=self.num_classes, img_index=img_index, 
                lam=lam, smoothing=self.smoothing)
        
            patch_output_type = 'seq'
            
        else:
            # Image-level mixup
            # reference: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py
            lam = np.random.beta(0.8, 0.8) # following DeiT
            samples_flipped = samples[img_index].mul_(1. - lam)
            samples.mul_(lam).add_(samples_flipped)
            
            targets = mixup_target(targets, num_classes=self.num_classes, img_index=img_index, 
                lam=lam, smoothing=self.smoothing)
            
            B = targets.shape[0]
            patch_targets = targets.reshape(B, 1, -1)
            patch_targets = patch_targets.repeat(1, self.patch_len * self.patch_len, 1)
        
            patch_output_type = 'avg'
        '''
        
        #return samples, targets, patch_targets, patch_output_type
        

if __name__ == '__main__':
    
    torch.manual_seed(10)
    np.random.seed(0)
    
    _B = 4
    _H, _W = 8, 8
    _patch_len = 4
    _num_classes = 4
    
    inputs = torch.zeros((_B, 1, _H, _W)).cuda()
    #inputs[0, ::] = 0
    #inputs[1, ::] = 1
    #inputs[2, ::] = 2
    #inputs[3, ::] = 3
    for i in range(inputs.shape[2]):
        for j in range(inputs.shape[3]):
            inputs[0, :, i, j] = i * inputs.shape[3] + j + 1
    inputs[1, ::] = inputs[0, ::] * 10
    inputs[2, ::] = inputs[0, ::] * 100
    inputs[3, ::] = inputs[0, ::] * 1000
    
    targets = torch.zeros((_B, 1)).long().cuda()
    targets[0, :] = 0
    targets[1, :] = 1
    targets[2, :] = 2
    targets[3, :] = 3
    
    print(inputs.long())
    
    mixup_fn = SwitchTokenMix(patch_len=_patch_len, 
        num_classes=_num_classes, smoothing=0.0)
    
    inputs, targets, patch_targets, _ = mixup_fn(inputs, targets)
    
    print(inputs.long())
    print(targets)
    print(patch_targets)
    
    #print(inputs.flatten(2).transpose(1, 2))
    