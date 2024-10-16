import platform
import pickle
from glob import glob
from PIL import Image
import numpy as np
import cv2
from timeit import default_timer as timer
import onnxruntime as ort
import torch
import os
import tensorrt
os.environ['LD_LIBRARY_PATH'] = tensorrt.__path__[0] # It may be needed depending on the system.
from models.clipseg import CLIPDensePredT 
from torchvision import transforms
from models.utils import convert2mask, ConfusionMatrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device = {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
print(platform.uname())

# Test Methods: 
# img_rand_noise: Single img. w/ rand. noise (100 trials)
# imgs: 500 img_rand_noise.
method = 'imgs'

img_size = 352
time_spent = []
time_spent_transform = []
logitsCLIPSegModTRT_IO = []
blur_kernel_size = 15

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225]), # std. ImageNet stats.
    transforms.Resize((img_size, img_size), antialias = True),
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

# Modified CLIPSeg with TRT. & IO binding
with open("cond/conditionals_processed.pkl", 'rb') as f:
    conditionals = pickle.load(f)

conditionals_0_cuda = []
for c in conditionals[0]:
    tmp = ort.OrtValue.ortvalue_from_numpy(c.half().unsqueeze(0).cpu().numpy(), device_type = device)
    conditionals_0_cuda.append(tmp)

conditionals_1_cuda = []
for c in conditionals[1]:
    tmp = ort.OrtValue.ortvalue_from_numpy(c.half().unsqueeze(0).cpu().numpy(), device_type = device)
    conditionals_1_cuda.append(tmp)
 
def seg_mod_trt_IO(input_img, img):
    loop_start = timer()
    
    # Start model computations
    img_tensor = transform(input_img).unsqueeze(0).half().cpu().numpy()
    time_spent_transform.append(timer() - loop_start)
    
    io_binding_0.bind_cpu_input('input_tensor', img_tensor) # Because in the real scenario this will happen
    ort_session_0.run_with_iobinding(io_binding_0)

    logits = []
    for c0, c1 in zip(conditionals_0_cuda, conditionals_1_cuda):
        io_binding_1.bind_input(name = 'cond0', 
                                device_type = c0.device_name(), 
                                device_id = 0, 
                                element_type = np.float16, 
                                shape = c0.shape(), 
                                buffer_ptr = c0.data_ptr())
        io_binding_1.bind_input(name = 'cond1', 
                                device_type = c1.device_name(), 
                                device_id = 0, 
                                element_type = np.float16, 
                                shape = c1.shape(), 
                                buffer_ptr = c1.data_ptr())
        ort_session_1.run_with_iobinding(io_binding_1)
        segmentation = io_binding_1.copy_outputs_to_cpu()[0]
        logits.append(segmentation)

    logits = torch.squeeze(torch.Tensor(np.array(logits)), 1)
   
    # Softmax
    logits = logits.softmax(dim = 0).cpu().numpy()
  
    # Keep only the positive prompts
    logits = logits[-len(positive_prompts):].sum(axis = 0)[0]
   
    # Blur to smooth the ViT patches
    logits = cv2.blur(logits, (blur_kernel_size, blur_kernel_size))
 
    # Convert to int8
    logits = (logits * 255).astype('uint8')      
       
    logitsCLIPSegModTRT_IO.append(logits) 
  
    # End model computations
    time_spent.append(timer() - loop_start)

    segmentation_file = f"fp16_quant_dataset/segmentations/Mod_trt_IO/segmentation_Original_Mod_TRT_IO_{str(img).split('/')[-1].split('.')[0].split('-')[-1]}.png"
    print(f"Saving {segmentation_file}...")
    Image.fromarray(logits).save(segmentation_file)    

# Model setup  
# https://onnxruntime.ai/docs/api/python/api_summary.html#data-inputs-and-outputs
a3cuda = ort.OrtValue.ortvalue_from_shape_and_type(shape = (485, 1, 768), 
                                                   element_type = np.float16,
                                                   device_type = device)
a6cuda = ort.OrtValue.ortvalue_from_shape_and_type(shape = (485, 1, 768), 
                                                   element_type = np.float16,
                                                   device_type = device)
a9cuda = ort.OrtValue.ortvalue_from_shape_and_type(shape = (485, 1, 768), 
                                                   element_type = np.float16,
                                                   device_type = device)

providers = [('TensorrtExecutionProvider',
             {'trt_engine_cache_enable': True,
              'trt_engine_cache_path': ".",
              'trt_builder_optimization_level': 5,
              'trt_fp16_enable': True,
              'trt_int8_enable': False}), 
              'CUDAExecutionProvider']
             
sess_options = ort.SessionOptions()
ort_session_0 = ort.InferenceSession("onnx/CLIPActivations_fp16_352_simpl.onnx", sess_options = sess_options, providers = providers)

sess_options = ort.SessionOptions()
ort_session_1 = ort.InferenceSession("onnx/CLIPSegDecoder_fp16_352_simpl.onnx", sess_options = sess_options, providers = providers)

io_binding_0 = ort_session_0.io_binding()
io_binding_0.bind_output(name = 'activations3', 
                         device_type =a3cuda.device_name(), 
                         device_id = 0, 
                         element_type = np.float16, 
                         shape = a3cuda.shape(), 
                         buffer_ptr = a3cuda.data_ptr())
io_binding_0.bind_output(name = 'activations6', 
                         device_type = a6cuda.device_name(), 
                         device_id = 0, 
                         element_type = np.float16, 
                         shape = a6cuda.shape(), 
                         buffer_ptr = a6cuda.data_ptr())
io_binding_0.bind_output(name = 'activations9', 
                         device_type = a9cuda.device_name(), 
                         device_id = 0, 
                         element_type = np.float16, 
                         shape = a9cuda.shape(), 
                         buffer_ptr = a9cuda.data_ptr())

io_binding_1 = ort_session_1.io_binding()
io_binding_1.bind_input(name = 'activations3', 
                         device_type = a3cuda.device_name(), 
                         device_id = 0, 
                         element_type = np.float16, 
                         shape = a3cuda.shape(), 
                         buffer_ptr = a3cuda.data_ptr())
io_binding_1.bind_input(name = 'activations6', 
                         device_type = a6cuda.device_name(), 
                         device_id = 0, 
                         element_type = np.float16, 
                         shape = a6cuda.shape(), 
                         buffer_ptr = a6cuda.data_ptr())
io_binding_1.bind_input(name = 'activations9', 
                         device_type = a9cuda.device_name(), 
                         device_id = 0, 
                         element_type = np.float16, 
                         shape = a9cuda.shape(), 
                         buffer_ptr = a9cuda.data_ptr())
io_binding_1.bind_output('segmentation')

# Warm up
img_tensor = transform(input_img).unsqueeze(0).half().cpu().numpy()
io_binding_0.bind_cpu_input('input_tensor', img_tensor) # Because in the real scenario, this will happen
ort_session_0.run_with_iobinding(io_binding_0)

logits = []
for c0, c1 in zip(conditionals_0_cuda, conditionals_1_cuda):
    io_binding_1.bind_input(name = 'cond0', 
                            device_type = c0.device_name(), 
                            device_id = 0, 
                            element_type = np.float16, 
                            shape = c0.shape(), 
                            buffer_ptr = c0.data_ptr())
    io_binding_1.bind_input(name = 'cond1', 
                            device_type = c1.device_name(), 
                            device_id = 0, 
                            element_type = np.float16, 
                            shape = c1.shape(), 
                            buffer_ptr = c1.data_ptr())
    
    ort_session_1.run_with_iobinding(io_binding_1)
    segmentation = io_binding_1.copy_outputs_to_cpu()[0]
        
    logits.append(segmentation)
        
if method == 'img_rand_noise':
    # Test on single ex. img. w/ rand. noise
    total_trials = 100
    start = timer()
    for trial in range(total_trials):
        input_img_rand = Image.fromarray((np.asanyarray(input_img) + np.random.rand(352, 352, 3) * 2).astype(np.uint8))
        seg_mod_trt_IO(input_img_rand, trial)
elif method == 'imgs':
    # Test w/ fp16_quant ds.
    images = glob('fp16_quant_dataset/input_images/*.png')
    total_trials = len(images)   
    start = timer()
    for img in images:
        input_img_pil = Image.open(img).resize((img_size, img_size))
        seg_mod_trt_IO(input_img_pil, img)
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
    
        inp_image = inp_image.cuda()
        logits = model32(inp_image, conditional = prompts, return_features = False, mask = None)[0]
 
    # Softmax
    logits = logits.softmax(dim = 0).detach().cpu().numpy()

    # Keep only the positive prompts
    logits = logits[-len(positive_prompts):].sum(axis = 0)[0]
   
    # Blur to smooth the ViT patches
    logits = cv2.blur(logits, (blur_kernel_size, blur_kernel_size))

    # Convert to int8
    logits = (logits * 255).astype('uint8')  

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
        
# Accuracy comparison: Original DOVESEI FP32 vs CLIPSeg Mod TRT IO.
confmat = ConfusionMatrix(2) # background:0 and safe landing:1

for gt, test in zip(logits32, logitsCLIPSegModTRT_IO):
    gt_mask = torch.from_numpy(convert2mask(gt))
    test_mask = torch.from_numpy(convert2mask(test))

    confmat.update(gt_mask.flatten(),
                   test_mask.flatten())

mean_acc, acc, iou = confmat.compute()
print(confmat)
print(f"mean_acc: {mean_acc.item()}, acc: {acc.cpu().numpy()}, iou: {iou.cpu().numpy()}")
