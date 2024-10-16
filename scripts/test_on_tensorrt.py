from glob import glob
import platform
import pickle
import os
from timeit import default_timer as timer
import numpy as np
from PIL import Image
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from models.clipseg import CLIPDensePredT 
import cv2
from torchvision import transforms
from models.utils import convert2mask, ConfusionMatrix

os.environ['LD_LIBRARY_PATH'] = trt.__path__[0] # It may be needed depending on the system.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device = {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
print(platform.uname())

# Test Methods: 
# img_rand_noise: Single img. w/ rand. noise (100 trials)
# imgs: 500 imgs.
method = 'img_rand_noise'

# Values according to DOVESEI's defaults
img_size = 352
time_spent = []
time_spent_transform = []
logitsCLIPSegModTRT_IO_trtexec = []
blur_kernel_size = 15

print(f"Image size: {img_size, img_size}")
print(f"Blur kernel size: {blur_kernel_size}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225]), # std. ImageNet stats.
    transforms.Resize((img_size, img_size), antialias = True)])  
    
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

with open("cond/conditionals_processed.pkl", 'rb') as f:
    conditionals = pickle.load(f) 
cond0 = conditionals[0].half().cpu().numpy()
cond1 = conditionals[1].half().cpu().numpy()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

def seg(img, trial):
    loop_start = timer()
    input_img = (np.asarray(img).transpose((2, 0, 1)) / 255).astype('float32')
    time_spent_transform.append(timer() - loop_start)
    logits = predict(np.expand_dims(input_img, axis = 0))
    
    #logits = torch.Tensor(logits).softmax(dim = 0).cpu().numpy()
    
    # Keep only the positive prompts
    logits = logits[-len(positive_prompts):].sum(axis = 0)[0]

    # Blur to smooth the ViT patches
    logits = cv2.blur(logits, (blur_kernel_size, blur_kernel_size))

    # Converts to uint8
    logits = np.array(logits * 255).astype('uint8')

    logitsCLIPSegModTRT_IO_trtexec.append(logits)
    
    # End model computations
    time_spent.append(timer() - loop_start)
    
    segmentation_file = f"int8_quant_dataset/segmentations/segmentation_trtexec_{str(trial).split('/')[-1].split('.')[0].split('-')[-1]}.png"                          
    print(f"Saving {segmentation_file}...")
    Image.fromarray(logits).save(segmentation_file)
    
print("Preparing CLIPActivations...")
# CLIPActivations setup
CLIPActivations_engine = "onnx/CLIPActivations16.trt"
with open(CLIPActivations_engine, 'rb') as f:
    engine_bytes = f.read()
    activations_engine = runtime.deserialize_cuda_engine(engine_bytes)

activations_context = activations_engine.create_execution_context()

activations_input_binding_idx = activations_engine.get_binding_index('input_tensor')
activations_output_binding_idx_0 = activations_engine.get_binding_index('activations3')
activations_output_binding_idx_1 = activations_engine.get_binding_index('activations6')
activations_output_binding_idx_2 = activations_engine.get_binding_index('activations9')

activations_input_shape = (1, 3, img_size, img_size)

activations_context.set_binding_shape(activations_input_binding_idx, activations_input_shape)

activations_output_0 = np.empty((485, 1, 768), dtype = np.float16)
activations_output_1 = np.empty((485, 1, 768), dtype = np.float16)
activations_output_2 = np.empty((485, 1, 768), dtype = np.float16)
activations_input = np.empty((1, 3, img_size, img_size), dtype = np.float32)

# Allocate device memory
activations_d_input = cuda.mem_alloc(1 * activations_input.nbytes)

activations_d_output_0 = cuda.mem_alloc(1 * activations_output_0.nbytes)
activations_d_output_1 = cuda.mem_alloc(1 * activations_output_1.nbytes)
activations_d_output_2 = cuda.mem_alloc(1 * activations_output_2.nbytes)

activations_bindings = [None] * 4
activations_bindings[activations_input_binding_idx] = int(activations_d_input)
activations_bindings[activations_output_binding_idx_0] = int(activations_d_output_0)
activations_bindings[activations_output_binding_idx_1] = int(activations_d_output_1)
activations_bindings[activations_output_binding_idx_2] = int(activations_d_output_2)

print("Preparing CLIPSegDecoder...")
# CLIPSegDecoder setup
CLIPSegDecoder_engine = "onnx/CLIPSegDecoder16NoSoft.trt"
with open(CLIPSegDecoder_engine, 'rb') as f:
    engine_bytes = f.read()
    decoder_engine = runtime.deserialize_cuda_engine(engine_bytes)

decoder_context = decoder_engine.create_execution_context()

decoder_input_binding_idx_0 = decoder_engine.get_binding_index('cond0')
decoder_input_binding_idx_1 = decoder_engine.get_binding_index('cond1')
decoder_input_binding_idx_2 = decoder_engine.get_binding_index('activations3')
decoder_input_binding_idx_3 = decoder_engine.get_binding_index('activations6')
decoder_input_binding_idx_4 = decoder_engine.get_binding_index('activations9')
decoder_output_binding_idx_0 = decoder_engine.get_binding_index('segmentation')

decoder_context.set_binding_shape(decoder_input_binding_idx_0, (13, 64))
decoder_context.set_binding_shape(decoder_input_binding_idx_1, (13, 64))
decoder_context.set_binding_shape(decoder_input_binding_idx_2, (485, 1, 768))
decoder_context.set_binding_shape(decoder_input_binding_idx_3, (485, 1, 768))
decoder_context.set_binding_shape(decoder_input_binding_idx_4, (485, 1, 768))

decoder_output_0 = np.empty((13 , 1, 352, 352), dtype = np.float32)

# Allocate device memory
decoder_d_input_0 = cuda.mem_alloc(1 * cond0.nbytes)
decoder_d_input_1 = cuda.mem_alloc(1 * cond1.nbytes)

decoder_d_output_0 = cuda.mem_alloc(1 * decoder_output_0.nbytes)

decoder_bindings = [None] * 6
decoder_bindings[decoder_input_binding_idx_0] = int(decoder_d_input_0)
decoder_bindings[decoder_input_binding_idx_1] = int(decoder_d_input_1)
decoder_bindings[decoder_input_binding_idx_2] = int(activations_d_output_0)
decoder_bindings[decoder_input_binding_idx_3] = int(activations_d_output_1)
decoder_bindings[decoder_input_binding_idx_4] = int(activations_d_output_2)

decoder_bindings[decoder_output_binding_idx_0] = int(decoder_d_output_0)

# Use the models
stream = cuda.Stream()

# Transfer conditionals to device
cuda.memcpy_htod_async(decoder_d_input_0, np.ascontiguousarray(cond0), stream)
cuda.memcpy_htod_async(decoder_d_input_1, np.ascontiguousarray(cond1), stream)

# Result gets copied into output
def predict(input_data):
    # Transfer input data to device
    cuda.memcpy_htod_async(activations_d_input, np.ascontiguousarray(input_data), stream)
    
    # Execute model
    activations_context.execute_async_v2(activations_bindings, stream.handle)
    
    # transfer predictions back
    cuda.memcpy_dtoh_async(activations_output_0, activations_d_output_0, stream)
    cuda.memcpy_dtoh_async(activations_output_1, activations_d_output_1, stream)
    cuda.memcpy_dtoh_async(activations_output_2, activations_d_output_2, stream)
    
    # Syncronize threads
    stream.synchronize()
    
    # Execute model
    decoder_context.execute_async_v2(decoder_bindings, stream.handle)
    
    # Transfer predictions back
    cuda.memcpy_dtoh_async(decoder_output_0, decoder_d_output_0, stream)
    
    # Syncronize threads
    stream.synchronize()
    
    return decoder_output_0

# Warm up
print("Warm up...")
_ = predict(np.zeros((1, 3, 352, 352), dtype = 'float32'))
        
if method == 'img_rand_noise':
    # Test on single ex. img. w/ rand. noise
    total_trials = 50
    start = timer()
    for trial in range(total_trials):
        input_img_rand = Image.fromarray((np.asanyarray(input_img) + np.random.rand(352, 352, 3) * 2).astype(np.uint8))
        seg(input_img_rand, trial)
elif method == 'imgs':
    # Test w/ int8_quant ds.
    images = glob('int8_quant_dataset/input_images/*.png')
    total_trials = len(images)
    start = timer()
    for img in images:
        input_img_pil = Image.open(img).resize((img_size, img_size))
        seg(input_img_pil, img)
else:
    print("Invalid test method") 

print(f"Total time: {timer() - start:.6f}s")
print(f"Device: {device}, Total trials: {total_trials}, Mean: {np.asarray(time_spent).mean():0.6}s, Std: {np.asarray(time_spent).std():0.6f}s")
print(f"Mostly image transformation - Mean: {np.asarray(time_spent_transform).mean():0.6f}s, Std: {np.asarray(time_spent_transform).std():0.6f}s")

# Float32       
def seg32(input_img, img):
    loop_start = timer()

    # Start model computations
    with torch.no_grad():
        inp_image = transform(input_img).unsqueeze(0)
        inp_image = torch.cat([inp_image] * len(prompts))
   
        time_spent_transform.append(timer() - loop_start)
    
        inp_image = inp_image.cuda()
        logits = model32(inp_image, conditional = prompts, return_features = False, mask = None)[0]
        
    # Softmax
    logits = logits.softmax(dim = 0).detach().cpu().numpy()       
    
    # Keep only the positive prompts
    logits = logits[-len(positive_prompts):].sum(axis = 0)[0]

    # Blur to smooth the ViT patches
    logits = cv2.blur(logits, (blur_kernel_size, blur_kernel_size))

    # Converts to uint8
    logits = np.array(logits * 255).astype('uint8')
    
    logits32.append(logits)
        
model32 = CLIPDensePredT(version = 'ViT-B/16', reduce_dim = 64, complex_trans_conv = True, device = 'cuda', openclip = False)
model32.load_state_dict(torch.load('weights/rd64-uni-refined.pth'), strict = False)
model32.eval()
model32.cuda()
model32.float()

# Warm up
with torch.no_grad():
    inp_image = transform(input_img).unsqueeze(0)
    inp_image = torch.cat([inp_image] * len(prompts))
    inp_image = inp_image.cuda()
    logits = model32(inp_image, conditional = prompts, return_features = False, mask = None)[0]

logits32 = []
if method == 'img_rand_noise':
    for trial in range(total_trials):
        input_img_rand = Image.fromarray((np.asanyarray(input_img) + np.random.rand(352, 352, 3) * 2).astype(np.uint8))
        seg32(input_img_rand, trial)
elif method == 'imgs':
    for img in images:
        input_img_pil = Image.open(img).resize((img_size, img_size))
        seg32(input_img_pil, img)
else:
    print("Invalid test method") 
    
# Accuracy comparison: Original DOVESEI FP32 vs CLIPSeg Mod TRT IO Norm. softmax.
confmat = ConfusionMatrix(2) # background:0 and safe landing:1

for gt, test in zip(logits32, logitsCLIPSegModTRT_IO_trtexec):
    gt_mask = torch.from_numpy(convert2mask(gt))
    test_mask = torch.from_numpy(convert2mask(test))

    confmat.update(gt_mask.flatten(),
                   test_mask.flatten())

mean_acc, acc, iou = confmat.compute()
print(confmat)
print(f"mean_acc: {mean_acc.item()}, acc: {acc.cpu().numpy()}, iou: {iou.cpu().numpy()}")
