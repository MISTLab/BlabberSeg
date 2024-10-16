import platform
import pickle
from glob import glob
from PIL import Image
import numpy as np
import cv2
from timeit import default_timer as timer
from torchvision import transforms
from models.utils import convert2mask, ConfusionMatrix
import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from models.clipseg import CLIPDensePredT 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device = {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
print(platform.uname())

# Test Methods: 
# img_rand_noise: Single img. w/ rand. noise (100 trials)
# imgs: 500 imgs.
method = 'imgs'

img_size = 352
time_spent = []
time_spent_transform = []
logitsHF_Reusing = []
blur_kernel_size = 15
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225]),
    transforms.Resize((img_size, img_size)),
])

input_img = Image.open("example_earth_200m.jpg").resize((img_size, img_size))

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

# HuggingFace + Reusing model
with open("cond/conditionals_FP32.pkl", 'rb') as f:
    cond_from_clip_float = pickle.load(f).float()
    
def segHF_Reusing(input_img, img):
    loop_start = timer()
    
    # Start model computations
    with torch.no_grad():
        inputs = processor(images = [input_img] * len(prompts), padding = True, return_tensors = "pt")
        time_spent_transform.append(timer() - loop_start)
        inputs['conditional_embeddings'] = cond_from_clip_float
        for k in inputs:
            if torch.cuda.is_available():
                inputs[k] = inputs[k].cuda()
        logits = modelHF_Reusing(**inputs).logits
      
    # Softmax
    logits = np.expand_dims(logits.softmax(dim = 0).detach().cpu().numpy(), axis = 1)

    # Keep only the positive prompts
    logits = logits[-len(positive_prompts):].sum(axis = 0)[0]
    
    # Blur to smooth the ViT patches
    logits = cv2.blur(logits, (blur_kernel_size, blur_kernel_size))

    # Convert to int8
    logits = (logits * 255).astype('uint8')

    logitsHF_Reusing.append(logits)
              
    # End model computations
    time_spent.append(timer() - loop_start)
    
    segmentation_file = f"fp16_quant_dataset/segmentations/HF_Reusing/segmentation_Original_CLIPSeg_HF_Reusing_{img.split('/')[-1].split('.')[0].split('-')[-1]}.png"
    print(f"Saving {segmentation_file}...")
    Image.fromarray(logits).save(segmentation_file)

# Model setup
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined") 
if 'cuda' in device:
    modelHF_Reusing = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").cuda()
else:
    modelHF_Reusing = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
modelHF_Reusing = modelHF_Reusing.eval()

# Warm up
with torch.no_grad():
    inputs = processor(images = [input_img] * len(prompts), padding = True, return_tensors = "pt")
    inputs['conditional_embeddings'] = cond_from_clip_float
    for k in inputs:
        if torch.cuda.is_available():
            inputs[k] = inputs[k].cuda()
    logits = modelHF_Reusing(**inputs).logits

if method == 'img_rand_noise':
    # Test on single ex. img. w/ rand. noise
    total_trials = 100

    start = timer()
    for trial in range(total_trials):
        input_img_rand = Image.fromarray((np.asanyarray(input_img) + np.random.rand(352, 352, 3) * 2).astype(np.uint8))
        segHF_Reusing(input_img_rand, trial)
elif method == 'imgs':
    # Test w/ fp16_quant ds.
    images = glob('fp16_quant_dataset/input_images/*.png')
    total_trials = len(images)   
    start = timer()
    for img in images:
        input_img_pil = Image.open(img).resize((img_size, img_size))
        segHF_Reusing(input_img_pil, img)
else:
    print("Invalid test method") 

print(f"Total time: {timer() - start:.6f}s")
print(f"Device: {device}, Total trials: {total_trials}, Mean: {np.asarray(time_spent).mean():0.6f}s, Std: {np.asarray(time_spent).std():0.6f}s")
print(f"Mostly image transformation - Mean: {np.asarray(time_spent_transform).mean():0.6f}s, Std: {np.asarray(time_spent_transform).std():0.6f}s")

# Float32
def seg32(input_img, img):
    loop_start = timer()

    # Start model computations
    with torch.no_grad():
        inp_image = transform(input_img).unsqueeze(0)
        inp_image = torch.cat([inp_image] * len(prompts))
   
        time_spent_transform.append(timer() - loop_start)
        if torch.cuda.is_available():    
            inp_image = inp_image.cuda()
        logits = model32(inp_image, conditional = prompts, return_features = False, mask = None)[0]
        
    # Softmax
    logits = logits.softmax(dim = 0).detach().cpu().numpy()
    
    # Keep only the positive prompts
    logits = logits[-len(positive_prompts):].sum(axis = 0)[0]
    
    # Blur to smooth the ViT patches
    logits = cv2.blur(logits,(blur_kernel_size, blur_kernel_size))
    
    # Converts to uint8
    logits = (logits * 255).astype('uint8')
    logits32.append(logits)
        
model32 = CLIPDensePredT(version = 'ViT-B/16', reduce_dim = 64, complex_trans_conv = True, device = torch.device(device), openclip = False)
model32.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location = torch.device(device)), strict = False)
model32.eval()
if torch.cuda.is_available():
    model32.cuda()
model32.float()

# Warm up
with torch.no_grad():
    inp_image = transform(input_img).unsqueeze(0)
    inp_image = torch.cat([inp_image] * len(prompts))
    if torch.cuda.is_available():
        inp_image = inp_image.cuda()
    logits = model32(inp_image, conditional = prompts, return_features = False, mask = None)[0]

logits32 = []
if method == 'img_rand_noise':
    # Test on single ex. img. w/ rand. noise
    start = timer()
    for trial in range(total_trials):
        input_img_rand = Image.fromarray((np.asanyarray(input_img) + np.random.rand(352, 352, 3) * 2).astype(np.uint8))
        seg32(input_img_rand, trial)
elif method == 'imgs':
    # Test w/ fp16_quant ds.
    start = timer()
    for img in images:
        input_img_pil = Image.open(img).resize((img_size, img_size))
        seg32(input_img_pil, img)
else:
    print("Invalid test method") 
        
# Accuracy comparison: Original DOVESEI FP32 vs Original DOVESEI HF Reusing.
confmat = ConfusionMatrix(2) # background:0 and safe landing:1

for gt, test in zip(logits32, logitsHF_Reusing):
    gt_mask = torch.from_numpy(convert2mask(gt))
    test_mask = torch.from_numpy(convert2mask(test))
    confmat.update(gt_mask.flatten(),
                   test_mask.flatten())

mean_acc, acc, iou = confmat.compute()
print(confmat)
print(f"mean_acc: {mean_acc.item()}, acc: {acc.cpu().numpy()}, iou: {iou.cpu().numpy()}")
