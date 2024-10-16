# Save the models with fixed and custom rescaled_pos_emb and class_embedding_reshaped
# You only need to run it once and then use the saved files later on
# - It depends on OpenCLIP: https://github.com/mlfoundations/open_clip
#      - `pip install open_clip_torch`

# # Custom params generator
# # It generates the models for each image size (112 x 112, 224 x 224, 352 x 352) and saves in the current directory.

from collections import OrderedDict
import torch
from models.clipseg_mod_gen import CLIPActivationsGen # Used to generate rescaled_pos_emb and class_embedding_reshaped.

# Float16 versions
for img_size in 112, 224, 352:
     activations_model = CLIPActivationsGen(img_size = img_size)
     custom_params = OrderedDict([('rescaled_pos_emb', activations_model.half().state_dict()['rescaled_pos_emb']),
                                  ('class_embedding_reshaped', activations_model.half().state_dict()['class_embedding_reshaped'])])
                
     torch.save(custom_params, f"CLIPActivations/CLIPActivations_float16_{img_size}.pth")

# It saves only what CLIPSeg needs from CLIP 
# (the parts that don't need to be customized according to the input image)
activations_model.half()
only_clip = activations_model.state_dict()

# Delete the parameters that depend on image size to avoid having problems with the order the parameters are loaded from file.
del only_clip['rescaled_pos_emb']
del only_clip['class_embedding_reshaped']
del only_clip['positional_embedding'] # Not needed anymore
torch.save(only_clip, f"CLIPActivations/CLIP_float16.pth")

# Float32 versions
# Custom params generator
# It generates the models for each image size (112 x 112, 224 x 224, 352 x 352) and saves in the current directory.
for img_size in 112, 224, 352:
    activations_model = CLIPActivationsGen(img_size = img_size)
    custom_params = OrderedDict([('rescaled_pos_emb', activations_model.float().state_dict()['rescaled_pos_emb']),
                                 ('class_embedding_reshaped', activations_model.float().state_dict()['class_embedding_reshaped'])])
                
    torch.save(custom_params, f"CLIPActivations/CLIPActivations_float32_{img_size}.pth")

# It saves only what CLIPSeg needs from CLIP 
# (the parts that don't need to be customized according to the input image)
activations_model.float()
only_clip = activations_model.state_dict()

# Delete the parameters that depend on image size to avoid having problems with the order the parameters are loaded from file.
del only_clip['rescaled_pos_emb']
del only_clip['class_embedding_reshaped']
del only_clip['positional_embedding'] # Not needed anymore.
torch.save(only_clip, f"CLIPActivations/CLIP_float32.pth")
