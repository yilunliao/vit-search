'''
    Implementation of masked layer norm.
    Handle different effective channel sizes for different examples within a batch.

    Input to masked layer norm is of shape B x N x C.

    Reference : 
        1. MABN (forward, backward): https://github.com/megvii-model/MABN/blob/master/MABN.py
        2. Pytorch LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        3. 
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLayerNormFunc(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x, weight, bias, eps, mask):
        '''
            x: shape is B x N x C
            mask: shape is B x 1 x C and is of boolean type.
        '''
        ctx.eps = eps
        
        B, N, C = x.size()
        unmask_percent = mask.float()
        # calculate percentage of unmasked (effective) channels 
        unmask_percent = torch.mean(unmask_percent, dim=2, keepdim=True) 
        inv_unmask_percent = 1.0 / unmask_percent

        x_mu = x.mean(dim=2, keepdim=True)
        x_mu = x_mu * inv_unmask_percent
        x2 = (x ** 2).mean(dim=2, keepdim=True)
        x2 = x2 * inv_unmask_percent
        var = x2 - (x_mu ** 2)
        
        inv_std = 1.0 / (var + eps).sqrt()
        z = (x - x_mu) * inv_std

        y = weight.view(1, 1, C) * z + bias.view(1, 1, C)
        #y = y * mask

        ctx.save_for_backward(inv_std, z, weight, inv_unmask_percent)

        return y
        

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):
        eps = ctx.eps
        
        B, N, C = grad_output.size()
        inv_std, z, weight, inv_unmask_percent = ctx.saved_variables 

        '''
            For gx
        '''
        dz = grad_output * weight.view(1, 1, C)
        # second term
        #mean_dz = dz.mean(dim=2, keepdim=True)
        #mean_dz = mean_dz * inv_unmask_percent
        mean_dz = dz #* inv_unmask_percent
        mean_dz = mean_dz.mean(dim=2, keepdim=True)
        # third term
        zdz = z * dz #* inv_unmask_percent
        mean_zdz = zdz.mean(dim=2, keepdim=True)
        #mean_zdz = mean_zdz 
        z_mean_zdz = z * mean_zdz

        gx = (- (mean_dz + z_mean_zdz)*inv_unmask_percent + dz) * inv_std        

        '''
            For g_gamma
        '''
        g_gamma = (grad_output * z).sum(dim=1).sum(dim=0)

        '''
            For g_beta
        '''
        g_beta = grad_output.sum(dim=1).sum(dim=0)

        return gx, g_gamma, g_beta, None, None
        

class MaskedLayerNorm(nn.Module):
    """
    Args:
        num_channels: assume input is of shape (B, N, C). Normalize over the last dim
            as used in vision transformer.
        eps: same as nn.LayerNorm.        
        
        always use affine transformation.
    """
    def __init__(self, num_channels, eps=1e-6):
        super(MaskedLayerNorm, self).__init__()
        
        self.register_parameter('weight', nn.Parameter(torch.ones(num_channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_channels)))
        
        self.eps = eps
        self.num_channels = num_channels
        
        # for the case of using pytorch layer_norm
        self.normalized_shape = tuple((num_channels, ))
        
        
    def forward(self, x, mask=None):
        '''
            x: shape is B x N x C
            mask: shape is B x 1 x C and is of boolean type.
        '''
        if mask is None:
            #B, N, C = x.size()
            #mask = torch.ones((B, 1, C), dtype=torch.bool).to(x.device)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            return x
        x = MaskedLayerNormFunc.apply(x, self.weight, self.bias, self.eps, mask)
        x = x * mask
        return x
    
    
    def extra_repr(self):
        return 'num_channels={}, eps={}'.format(self.num_channels, self.eps)

