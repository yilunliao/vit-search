import torch
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np
from timm.models.regnet import regnety_160
from timm.models.registry import register_model


class RegNetY160Upsample(nn.Module):
    
    def __init__(self, pretrained=False, **kwargs):
        super().__init__()
        self.model = regnety_160(pretrained=pretrained, **kwargs)
        self.output_size = 224
        
        
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        
        
    def forward(self, x):
        #temp = torch.nn.functional.interpolate(x, 
        #    size=(self.output_size, self.output_size), 
        #    mode='bicubic', 
        #    align_corners=True)
        temp = torchvision.transforms.functional.resize(x, size=(224, 224), 
            interpolation=Image.BICUBIC)
        out = self.model.forward(temp)
        return out
    

@register_model
def regnety_160_upsample(pretrained=False, **kwargs):
    '''
        Insert `network_def` to **kwargs in main.py
    '''
    model = RegNetY160Upsample(pretrained=pretrained, **kwargs)
    return model

