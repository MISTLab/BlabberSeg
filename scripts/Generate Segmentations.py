import platform
import pickle
from glob import glob

from PIL import Image
import numpy as np

import cv2

from timeit import default_timer as timer

import torch
from torchvision import transforms
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
print(platform.uname())

# https://github.com/MISTLab/DOVESEI/blob/main/src/ros2_open_voc_landing_heatmap/launch/start_aerialview.launch.py
negative_prompts = ["building, house, apartment-building, warehouse, shed, garage", 
                    "roof, rooftop, terrace, shelter, dome, canopy, ceiling", 
                    "tree, bare tree, tree during autumn, bush, tall-plant", 
                    "water, lake, river, swimming pool",
                    "people, crowd", 
                    "vehicle, car, train", 
                    "lamp-post, transmission-line", 
                    "fence, wall, hedgerow", 
                    "road, street, avenue, highway, drive, lane",
                    "stairs, steps, footsteps"]
positive_prompts = ["grass, dead grass, backyard, frontyard, courtyard, lawn", 
                    "sports-field, park, open-area, open-space, agricultural land",
                    "parking lot, sidewalk, gravel, dirt, sand, concrete floor, asphalt"] 

PROMPT_ENGINEERING = "aerial view, drone footage photo of {}, shade, shadows, low resolution"

prompts = [PROMPT_ENGINEERING.format(p) for p in negative_prompts]
prompts += [PROMPT_ENGINEERING.format(p) for p in positive_prompts]
print(f"Total number of prompts: {len(prompts)}")

# Values according to DOVESEI's defaults
img_size = 352
safety_threshold = .8
blur_kernel_size = 15
seg_dynamic_threshold = .3
DYNAMIC_THRESHOLD_MAXSTEPS = 100

print(f"Image size: {img_size,img_size}")
print(f"Safety_threshold: {safety_threshold}")
print(f"Blur kernel size: {blur_kernel_size}")
print(f"Segmentation dynamic threshold: {seg_dynamic_threshold}")
print(f"Segmentation dynamic threshold max steps: {DYNAMIC_THRESHOLD_MAXSTEPS}")
# -

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225]),
    transforms.Resize((img_size, img_size)),
])

# Model setup
from models.clipseg import CLIPDensePredT
print("Loading original CLIPSeg...")
model = CLIPDensePredT(version = 'ViT-B/16', reduce_dim = 64, complex_trans_conv = True, device = 'cuda', openclip = False)
model.load_state_dict(torch.load('weights/rd64-uni-refined.pth'), strict = False)
model.eval()
model.cuda()
model.float()

# ## Generate segmentations like DOVESEI
# * Fuse negative and positive prompts
# * Blur
# * Convert to uint8

for i, img in enumerate(glob("fp16_quant_dataset/input_images/*.png")):
    input_img_pil = Image.open(img).resize((img_size, img_size))
    with torch.no_grad():
        inp_image = transform(input_img_pil).unsqueeze(0)
        inp_image = torch.cat([inp_image] * len(prompts))
        inp_image = inp_image.cuda()
        logits = model(inp_image, conditional = prompts, return_features = False, mask = None)[0]
        logits = logits.softmax(dim = 0).detach().cpu().numpy()
        
    # Keep only the positive prompts
    logits = logits[-len(positive_prompts):].sum(axis = 0)[0]
    
    # Blur to smooth the ViT patches
    logits = cv2.blur(logits, (blur_kernel_size, blur_kernel_size))
    
    # Converts to uint8
    logits = (logits * 255).astype('uint8')
    Image.fromarray(logits).save("fp16_quant_dataset/segmentations/" + "seg_" + img.split('/')[-1])
    print(i, end = ", ")
