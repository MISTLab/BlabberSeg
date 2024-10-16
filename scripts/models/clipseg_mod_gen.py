"""
This modified version only works with the rd64-uni-refined.pth weights 
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O weights.zip
"""

from torch import nn
from copy import deepcopy
from models.clipseg_mod import CLIPActivationsBase
import open_clip

class CLIPActivationsGen(CLIPActivationsBase):
    def __init__(self, device = 'cpu', img_size = 352):
        super().__init__()
        
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', device = device, pretrained = 'openai')
        
        # Parameters from original CLIP ViT-B-16
        clip_input_size = 224
        self.width = clip_model.visual.conv1.out_channels
        patch_size = clip_model.visual.conv1.stride[0]
        heads = clip_model.visual.transformer.resblocks[0].attn.num_heads

        layers_needed = 10 # CLIPSeg only needs layers 0 to 9

        self.positional_embedding = deepcopy(clip_model.visual.positional_embedding)
        self.transformer = deepcopy(clip_model.visual.transformer)
        self.transformer.resblocks = self.transformer.resblocks[:layers_needed]
        self.conv1 = deepcopy(clip_model.visual.conv1)
        class_embedding = deepcopy(clip_model.visual.class_embedding).reshape(1, 1, 768)
        self.ln_pre = deepcopy(clip_model.visual.ln_pre)

        self.rescaled_pos_emb = nn.Parameter(self.rescale_pos_emb(img_size, 
                                                                  stride = patch_size, 
                                                                  token_shape = (clip_input_size // patch_size, clip_input_size // patch_size)))
        self.class_embedding_reshaped = nn.Parameter(class_embedding)

        head_dim = self.width // heads
        self.scaling = float(head_dim) ** -.5

        del clip_model

        for p in self.parameters():
            p.requires_grad_(False) # Whole model is non-trainable now
