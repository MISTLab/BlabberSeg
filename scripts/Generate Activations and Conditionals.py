import platform
import pickle
from glob import glob
from PIL import Image
import numpy as np
from timeit import default_timer as timer
from models.clipseg_mod import CLIPActivations, CLIPSegDecoder, CLIPSegDecoderProcessConditional
import torch
from torchvision import transforms
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device = {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
print(platform.uname())

img_size = 352

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225]), # std. ImageNet stats.
    transforms.Resize((img_size, img_size), antialias = True),
])

clip_model, preprocess = clip.load('ViT-B/16', device = device, jit = False)
clip_model.eval()
activations_model = CLIPActivations(img_size = img_size)

#FP32
clip_model.float()
process_conditional = CLIPSegDecoderProcessConditional()
process_conditional.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location = torch.device(device)), strict = False)
process_conditional.eval()
process_conditional.cuda()

# Pre-process prompts
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    cond32 = clip_model.encode_text(text_tokens)
    cond32_processed = process_conditional(cond32)
    
with open("cond/conditionals_FP32.pkl", 'wb') as f:
    pickle.dump(cond32, f)
    
with open("cond/conditionals_FP32_processed.pkl", 'wb') as f:
    pickle.dump(cond32_processed, f)
    
for p in process_conditional.parameters():
    print(p.dtype, end = ", ")

# Model setup
activations_model.load_state_dict(torch.load('CLIPActivations/CLIP_float32.pth', map_location = torch.device(device)), strict = False)
activations_model.load_state_dict(torch.load(f'CLIPActivations/CLIPActivations_float32_{img_size}.pth', map_location = torch.device(device)), strict = False)
activations_model.eval()
activations_model.cuda()

for i, img in enumerate(glob("fp16_quant_dataset/input_images/*.png")):
    input_img_pil = Image.open(img).resize((img_size, img_size))
    with torch.no_grad():
        inp_image = transform(input_img_pil).unsqueeze(0)
        inp_image = inp_image.cuda()
        activations = activations_model(inp_image)
        with open(f"fp16_quant_dataset/activations/activations32_{img.split('/')[-1].split('.')[0]}.pkl", 'wb') as f:
            pickle.dump(activations, f)
    print(i, end = ", ")
 
# FP16
# Model setup
activations_model.half()
activations_model.load_state_dict(torch.load('CLIPActivations/CLIP_float16.pth', map_location = torch.device(device)), strict = False)
activations_model.load_state_dict(torch.load(f'CLIPActivations/CLIPActivations_float16_{img_size}.pth', map_location = torch.device(device)), strict = False)
activations_model.eval()
activations_model.cuda().half()

for i, img in enumerate(glob("fp16_quant_dataset/input_images/*.png")):
    input_img_pil = Image.open(img).resize((img_size, img_size))
    with torch.no_grad():
        inp_image = transform(input_img_pil).unsqueeze(0).half()
        inp_image = inp_image.cuda()
        activations = activations_model(inp_image)
        with open(f"fp16_quant_dataset/activations/activations16_{img.split('/')[-1].split('.')[0]}.pkl", 'wb') as f:
            pickle.dump(activations, f)
    print(i, end = ", ")
