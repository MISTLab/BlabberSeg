# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# For some reason the TensorRT engines are not reused in some situations, even when the model is exactly the same,  therefore they are re-created and that takes a long time.     
#
# Useful links:
# * https://pytorch.org/docs/stable/onnx_torchscript.html
# * https://github.com/daquexian/onnx-simplifier
# * https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
#      * https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#quantization-on-gpu
#          * Unlike the CPU Execution Provider, TensorRT takes in a full precision model and a calibration result for inputs.
# * https://onnxruntime.ai/docs/performance/model-optimizations/float16.html
# * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/calibrate.py
# * https://github.com/NVIDIA/TensorRT/tree/master/samples/sampleINT8
# * https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html
# * https://github.com/NVIDIA/TensorRT/tree/96e23978cd6e4a8fe869696d3d8ec2b47120629b/samples/python/int8_caffe_mnist
#     * https://github.com/NVIDIA/TensorRT/blob/96e23978cd6e4a8fe869696d3d8ec2b47120629b/samples/python/int8_caffe_mnist/calibrator.py
#     * This discussion shows how to do the calibration using TensorRT: https://github.com/NVIDIA/TensorRT/issues/3131

# +
import platform
import pickle
from glob import glob
import shutil
import os

from PIL import Image
import numpy as np
import cv2

import matplotlib.pyplot as plt

from timeit import default_timer as timer

from models.clipseg_mod import (CLIPActivations, CLIPSegDecoder, CLIPSegDecoderProcessConditional, 
                                CLIPActivationsNormInput, CLIPSegDecoderSoftmax)

import torch
from torchvision import transforms
import torch.onnx

import onnx

import onnxruntime as ort
from onnxconverter_common import float16
# -

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(f"Using device = {device}")

# +
if device == 'cuda':
    print(torch.cuda.get_device_name(0))

print(platform.uname())
# -

print(torch.__version__)

print(onnx.__version__)

import torchvision
print(torchvision.__version__)

# +
# Controls if we are using int8+fp16 or only fp16

test_int8 = False

# +
img_size = 352
safety_threshold = 0.8 # values according to DOVESEI's defaults
blur_kernel_size = 15 # values according to DOVESEI's defaults
seg_dynamic_threshold = 0.10 # values according to DOVESEI's defaults
DYNAMIC_THRESHOLD_MAXSTEPS = 100 # values according to DOVESEI's defaults
len_positive_prompts = 3

print(f"Image size: {img_size,img_size}")
print(f"Safety_threshold: {safety_threshold}")
print(f"Blur kernel size: {blur_kernel_size}")
print(f"Segmentation dynamic threshold: {seg_dynamic_threshold}")
print(f"Segmentation dynamic threshold max steps: {DYNAMIC_THRESHOLD_MAXSTEPS}")
# -

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # standard ImageNet stats
    transforms.Resize((img_size, img_size), antialias=True),
])

with open("int8_quant_dataset/conditionals.pkl", 'rb') as f:
    conditionals = pickle.load(f) # float32

# As far as I understood, the best way is to use base ONNX models with float32 precision as the following conversions seem to work better that way. On the other hand, a model that is float16 and receives float16 should spend less time moving inputs and outputs between CPU and GPU.

# +
# activations_model = CLIPActivations(img_size=img_size)
# activations_model.load_state_dict(torch.load('CLIP_float32.pth', map_location=torch.device(device)), strict=False);
# activations_model.load_state_dict(torch.load(f'CLIPActivations_float32_{img_size}.pth', map_location=torch.device(device)), strict=False);
# activations_model.eval();
# activations_model.cuda(); # float32
# -

# This model has the transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # standard ImageNet stats
# inside the model to avoid slow downs when torch is not available (e.g. Jetson Nano 2GB).
activations_model_norm = CLIPActivationsNormInput(img_size=img_size)
activations_model_norm.load_state_dict(torch.load('CLIP_float32.pth', map_location=torch.device(device)), strict=False);
activations_model_norm.load_state_dict(torch.load(f'CLIPActivations_float32_{img_size}.pth', map_location=torch.device(device)), strict=False);
activations_model_norm.eval();
activations_model_norm.cuda(); # float32

# +
# seg_model = CLIPSegDecoder(img_size=img_size, batch_size=13)
# seg_model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device(device)), strict=False);
# seg_model.eval();
# seg_model.cuda(); # float32
# -

seg_model = CLIPSegDecoderSoftmax(img_size=img_size, batch_size=13)
seg_model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location=torch.device(device)), strict=False);
seg_model.eval();
seg_model.cuda(); # float32

# +
# fake_input = torch.rand(1,3,352,352, dtype=torch.float32, device='cuda')
# # Export the model
# torch.onnx.export(activations_model.cuda(),        # model to be converted
#                   fake_input,                      # model input (or a tuple for multiple inputs)
#                   "CLIPActivations_fp32_352.onnx", # where to save the model (can be a file or file-like object)
#                   export_params=True,              # store the trained parameter weights inside the model file
#                   opset_version=16,                # the ONNX version to export the model to
#                   do_constant_folding=True,        # whether to execute constant folding for optimization
#                   input_names = ['input_tensor'],  # the model's input names
#                   output_names = ['activations3', 
#                                   'activations6', 
#                                   'activations9'], # the model's output names
#                   dynamic_axes = None
#                   );
# -

fake_input = torch.rand(1,3,352,352, dtype=torch.float32, device='cuda')
# Export the model
torch.onnx.export(activations_model_norm.cuda(),        # model to be converted
                  fake_input,                           # model input (or a tuple for multiple inputs)
                  "CLIPActivations_fp32_norm_352.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,                   # store the trained parameter weights inside the model file
                  opset_version=14,                     # the ONNX version to export the model to
                  do_constant_folding=True,             # whether to execute constant folding for optimization
                  input_names = ['input_tensor'],       # the model's input names
                  output_names = ['activations3', 
                                  'activations6', 
                                  'activations9'],      # the model's output names
                  dynamic_axes = None,
                  verbose = True
                  );

fake_cond = torch.rand(13,64).cuda()
fake_activs = torch.rand(3,485,1,768).cuda()
torch.onnx.export(seg_model.cuda(),                      # model to be converted
                  (fake_cond, fake_cond, *fake_activs),  # model input (or a tuple for multiple inputs)
                  "CLIPSegDecoder_fp32_soft_352.onnx",        # where to save the model (can be a file or file-like object)
                  export_params=True,                    # store the trained parameter weights inside the model file
                  opset_version=14,                      # the ONNX version to export the model to
                  do_constant_folding=True,              # whether to execute constant folding for optimization
                  input_names = ['cond0','cond1', 
                                 'activations3',
                                 'activations6',
                                 'activations9'],        # the model's input names
                  output_names = ['segmentation'],       # the model's output names
                  dynamic_axes = None,
                  )

onnx_model_activations = onnx.load(f"CLIPActivations_fp32_norm_352.onnx")
onnx.checker.check_model(onnx_model_activations, full_check=True)

onnx_model_decoder = onnx.load(f"CLIPSegDecoder_fp32_soft_352.onnx")
onnx.checker.check_model(onnx_model_decoder, full_check=True)

# # Quantization

from onnxruntime.quantization import CalibrationDataReader, create_calibrator, write_calibration_table, CalibrationMethod
import tensorrt
os.environ['LD_LIBRARY_PATH'] = tensorrt.__path__[0] # it may be needed depending on the system...


def convert_model_batch_to_dynamic(model_path):
    model = onnx.load(model_path)
    initializers =  [node.name for node in model.graph.initializer]
    inputs = []
    for node in model.graph.input:
        if node.name not in initializers:
            inputs.append(node)
    input_name = inputs[0].name
    shape = inputs[0].type.tensor_type.shape
    dim = shape.dim
    if not dim[0].dim_param:
        dim[0].dim_param = 'N'
        model = onnx.shape_inference.infer_shapes(model)
        model_name = model_path.split(".")
        model_path = model_name[0] + "_dynamic.onnx"
        onnx.save(model, model_path)
    return [model_path, input_name]


# ### CLIPActivations Quantization

class DataReaderCLIPActivations(CalibrationDataReader):
    def __init__(self,
                 image_folder,
                 start_index=0,
                 end_index=0,
                 stride=1,
                 input_name=None):
        '''
        :param image_folder: image dataset folder
        :param start_index: start index of images
        :param end_index: end index of images
        :param stride: image size of each data get 
        :param model_path: model name and path
        '''

        self.image_folder = image_folder
        self.image_list = glob(image_folder + "/*.png")
        self.datasize = 0
        self.start_index = start_index
        self.end_index = len(self.image_list) if end_index == 0 else end_index
        self.stride = stride if stride >= 1 else 1

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
        ])

        assert input_name
        self.input_name = input_name

    def get_dataset_size(self):
        return len(self.image_list)

    def get_next(self):
        if self.start_index < self.end_index:
            input_img_pil = Image.open(self.image_list[self.start_index])
            input_torch = self.transform(input_img_pil).unsqueeze(0)
            self.start_index += self.stride
            return {self.input_name: np.asarray(input_torch)}
        else:
            return None


# Convert static batch to dynamic batch
input_model = "CLIPActivations_fp32_norm_352.onnx"
# input_model = "CLIPActivations_fp32_norm_352_simpl.onnx"
[new_model_path, input_name] = convert_model_batch_to_dynamic(input_model)

new_model_path

# +
# Inside the directory below onnx will save all the needed files to deploy the model,
# including the cached TensorRT engines
assert os.path.isdir('CLIPActivations_TensorRT'), "You must create this directory!"

calibrator = create_calibrator(new_model_path, 
                               augmented_model_path="CLIPActivations_TensorRT/augmented_model.onnx",
                               calibrate_method=CalibrationMethod.MinMax) # it crashes (OOM) for anything else
calibrator.set_execution_providers(["CUDAExecutionProvider"])        
data_reader = DataReaderCLIPActivations("int8_quant_dataset/input_images/",
                         start_index=0,
                         end_index=499,
                         stride=1,
                         input_name=input_name
                        )
calibrator.collect_data(data_reader)
write_calibration_table({k:v.range_value for k,v in calibrator.compute_data().data.items()})
shutil.move("./calibration.cache", "CLIPActivations_TensorRT/calibration.cache");
shutil.move("./calibration.json", "CLIPActivations_TensorRT/calibration.json");
shutil.move("./calibration.flatbuffers", "CLIPActivations_TensorRT/calibration.flatbuffers");
# -

# #### Two options to test here

# ##### int8 (using the calibration above)
# Here the engine is not created, only during the first inference, and the first inference seems to always be slow

if test_int8:
    providers=[('TensorrtExecutionProvider',
                {'trt_engine_cache_enable':True,
                 'trt_engine_cache_path':"./CLIPActivations_TensorRT",
                 'trt_builder_optimization_level':5,
                 'trt_fp16_enable':True,
                 'trt_int8_enable':True,
                 'trt_int8_calibration_table_name': 'calibration.flatbuffers',
                 'trt_int8_use_native_calibration_table':False,
                 # 'trt_dla_enable': True,
                 # 'trt_dla_core':0
                }), 
               'CUDAExecutionProvider'
              ]
    sess_options = ort.SessionOptions()
    ort_session_0 = ort.InferenceSession("CLIPActivations_TensorRT/augmented_model.onnx", sess_options=sess_options, providers=providers)

# ##### fp16
# For the fp16 the engine is clearly generate the moment it runs the cell below and the inference is fast (even the first one)

if not test_int8:
    providers=[('TensorrtExecutionProvider',
                {'trt_engine_cache_enable':True,
                 'trt_engine_cache_path':"./CLIPActivations_TensorRT/",
                 'trt_builder_optimization_level':5,
                 'trt_fp16_enable':True,
                 'trt_int8_enable':False,
                 # 'trt_dla_enable': True,
                 # 'trt_dla_core':0
                }), 
               'CUDAExecutionProvider'
              ]
    sess_options = ort.SessionOptions()
    ort_session_0 = ort.InferenceSession("CLIPActivations_fp32_norm_352.onnx", sess_options=sess_options, providers=providers)

# #### The rest should work no matter the option above...

# https://onnxruntime.ai/docs/api/python/api_summary.html#data-inputs-and-outputs
a3cuda = ort.OrtValue.ortvalue_from_shape_and_type(shape=(485, 1, 768), 
                                                   element_type=np.float32,
                                                   device_type=device)
a6cuda = ort.OrtValue.ortvalue_from_shape_and_type(shape=(485, 1, 768), 
                                                   element_type=np.float32,
                                                   device_type=device)
a9cuda = ort.OrtValue.ortvalue_from_shape_and_type(shape=(485, 1, 768), 
                                                   element_type=np.float32,
                                                   device_type=device)

io_binding_0 = ort_session_0.io_binding()

io_binding_0.bind_output(name='activations3', 
                         device_type=a3cuda.device_name(), 
                         device_id=0, 
                         element_type=np.float32, 
                         shape=a3cuda.shape(), 
                         buffer_ptr=a3cuda.data_ptr())
io_binding_0.bind_output(name='activations6', 
                         device_type=a6cuda.device_name(), 
                         device_id=0, 
                         element_type=np.float32, 
                         shape=a6cuda.shape(), 
                         buffer_ptr=a6cuda.data_ptr())
io_binding_0.bind_output(name='activations9', 
                         device_type=a9cuda.device_name(), 
                         device_id=0, 
                         element_type=np.float32, 
                         shape=a9cuda.shape(), 
                         buffer_ptr=a9cuda.data_ptr())

img_number=200

input_img = (np.asarray(Image.open(f'int8_quant_dataset/input_images/representative_dataset-{img_number:03d}.png').resize((img_size, img_size))).transpose((2,0,1))/255).astype('float32')

io_binding_0.bind_cpu_input('input_tensor', np.expand_dims(input_img, axis=0)) # because in the real scenario this will happen

# int8: 45.4 ms ± 407 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# fp16: 3.93 ms ± 83.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
ort_session_0.run_with_iobinding(io_binding_0)

with open(f"int8_quant_dataset/activations/activations_representative_dataset-{img_number:03d}.pkl", 'rb') as f:
    activations = pickle.load(f)

# only fp16: 0.010746346
# fp16 and int8: 0.26915744
np.sqrt(((activations[0].cpu().numpy()-a3cuda.numpy())**2).mean())


# ### CLIPSegDecoder Quantization

class DataReaderCLIPSegDecoder(CalibrationDataReader):
    def __init__(self,
                 data_folder,
                 conditionals_path,
                 start_index=0,
                 end_index=0,
                 stride=1,
                 input_names=['cond0','cond1','activations3','activations6','activations9']):

        with open(conditionals_path, 'rb') as f:
            self.conditionals = pickle.load(f)
        self.folder_list = glob(data_folder + "/*.pkl")
        self.datasize = 0
        self.start_index = start_index
        self.end_index = len(self.folder_list) if end_index == 0 else end_index
        self.stride = stride if stride >= 1 else 1
        self.input_names = input_names


    def get_dataset_size(self):
        return len(self.folder_list)

    def get_next(self):
        if self.start_index < self.end_index:
            with open(self.folder_list[self.start_index], 'rb') as f:
                activations = pickle.load(f)
            self.start_index += self.stride
            return {name: input_torch.cpu().float().numpy() for name, input_torch in zip(self.input_names, 
                                                                                 list(conditionals)+activations)}
        else:
            return None

# Convert static batch to dynamic batch
input_model = "CLIPSegDecoder_fp32_soft_352.onnx"
# input_model = "CLIPSegDecoder_fp32_352_simpl.onnx"
[new_model_path, input_name] = convert_model_batch_to_dynamic(input_model)

# +
# Inside the directory below onnx will save all the needed files to deploy the model,
# including the cached TensorRT engines
assert os.path.isdir('CLIPSegDecoder_TensorRT'), "You must create this directory!"

calibrator = create_calibrator(new_model_path, 
                               augmented_model_path="CLIPSegDecoder_TensorRT/augmented_model.onnx",
                               calibrate_method=CalibrationMethod.MinMax) # it crashes (OOM) for anything else
calibrator.set_execution_providers(["CUDAExecutionProvider"])        
data_reader = DataReaderCLIPSegDecoder("int8_quant_dataset/activations/",
                                       conditionals_path="int8_quant_dataset/conditionals.pkl",
                                       start_index=0,
                                       end_index=499,
                                       stride=1
                                      )
calibrator.collect_data(data_reader)
write_calibration_table({k:v.range_value for k,v in calibrator.compute_data().data.items()})
shutil.move("./calibration.cache", "CLIPSegDecoder_TensorRT/calibration.cache");
shutil.move("./calibration.json", "CLIPSegDecoder_TensorRT/calibration.json");
shutil.move("./calibration.flatbuffers", "CLIPSegDecoder_TensorRT/calibration.flatbuffers");
# -

# #### Two options to test here

# ##### int8 (using the calibration above)
# Here the engine is not created, only during the first inference, and the first inference seems to always be slow

if test_int8:
    providers=[('TensorrtExecutionProvider',
                {'trt_engine_cache_enable':True,
                 'trt_engine_cache_path':"./CLIPSegDecoder_TensorRT",
                 'trt_builder_optimization_level':5,
                 'trt_fp16_enable':True,
                 'trt_int8_enable':True,
                 'trt_int8_calibration_table_name': 'calibration.flatbuffers',
                 'trt_int8_use_native_calibration_table':False,
                 # 'trt_dla_enable': True,
                 # 'trt_dla_core':0
                }), 
               'CUDAExecutionProvider'
              ]
    sess_options = ort.SessionOptions()
    ort_session_1 = ort.InferenceSession("CLIPSegDecoder_TensorRT/augmented_model.onnx", sess_options=sess_options, providers=providers)

# ##### fp16
# For the fp16 the engine is clearly generated the moment it runs the cell below and the inference is fast (even the first one)

if not test_int8:
    providers=[('TensorrtExecutionProvider',
                {'trt_engine_cache_enable':True,
                 'trt_engine_cache_path':"./CLIPSegDecoder_TensorRT",
                 'trt_builder_optimization_level':5,
                 'trt_fp16_enable':True,
                 'trt_int8_enable':False,
                 # 'trt_dla_enable': True,
                 # 'trt_dla_core':0
                }), 
               'CUDAExecutionProvider'
              ]
    sess_options = ort.SessionOptions()
    ort_session_1 = ort.InferenceSession("CLIPSegDecoder_fp32_soft_352.onnx", sess_options=sess_options, providers=providers)

# ## Testing the full model using ONNX Runtime and IO Binding

# +
# For deployment, the conditionals should be saved as numpy arrays instead of torch
# to avoid the need of having pytorch installed

# https://onnxruntime.ai/docs/api/python/api_summary.html#data-inputs-and-outputs
c0 = ort.OrtValue.ortvalue_from_numpy(conditionals[0].cpu().numpy(), device_type=device)
c1 = ort.OrtValue.ortvalue_from_numpy(conditionals[1].cpu().numpy(), device_type=device)
# -

io_binding_1 = ort_session_1.io_binding()
io_binding_1.bind_input(name='activations3', 
                         device_type=a3cuda.device_name(), 
                         device_id=0, 
                         element_type=np.float32, 
                         shape=a3cuda.shape(), 
                         buffer_ptr=a3cuda.data_ptr())
io_binding_1.bind_input(name='activations6', 
                         device_type=a6cuda.device_name(), 
                         device_id=0, 
                         element_type=np.float32, 
                         shape=a6cuda.shape(), 
                         buffer_ptr=a6cuda.data_ptr())
io_binding_1.bind_input(name='activations9', 
                         device_type=a9cuda.device_name(), 
                         device_id=0, 
                         element_type=np.float32, 
                         shape=a9cuda.shape(), 
                         buffer_ptr=a9cuda.data_ptr())

logit_output = torch.empty((13, 1, 352, 352), dtype=torch.float32, device=device)
io_binding_1.bind_output(name='segmentation', 
                         device_type=device, 
                         element_type=np.float32, 
                         shape=tuple(logit_output.shape), 
                         buffer_ptr=logit_output.data_ptr())

io_binding_1.bind_input(name='cond0', 
                        device_type=c0.device_name(), 
                        device_id=0, 
                        element_type=np.float32, 
                        shape=c0.shape(), 
                        buffer_ptr=c0.data_ptr())
io_binding_1.bind_input(name='cond1', 
                        device_type=c1.device_name(), 
                        device_id=0, 
                        element_type=np.float32, 
                        shape=c1.shape(), 
                        buffer_ptr=c1.data_ptr())

# only fp16: 1.68 ms ± 50.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
# fp16 and int8: 22 ms ± 131 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
ort_session_1.run_with_iobinding(io_binding_1)

# logit_output = logit_output.softmax(dim=0).detach().cpu().float().numpy()
logit_output = logit_output.detach().cpu().float().numpy()
# Keep only the positive prompts
logit_output = logit_output[-len_positive_prompts:].sum(axis=0)[0]
# Blur to smooth the ViT patches
logit_output = cv2.blur(logit_output,(blur_kernel_size, blur_kernel_size))
# Converts to uint8
logit_output = (logit_output*255).astype('uint8')

plt.imshow(logit_output);
plt.colorbar();

input_img_pil = Image.open(f'int8_quant_dataset/input_images/representative_dataset-{img_number:03d}.png').resize((img_size, img_size))
segment_img_pil = Image.open(f'int8_quant_dataset/segmentations/seg_representative_dataset-{img_number:03d}.png').resize((img_size, img_size))
segment_img_original = np.asarray(segment_img_pil)/255.0

input_img_pil

fig, axs = plt.subplots(1,2)
axs[0].imshow(logit_output, vmin=0, vmax=segment_img_original.max()*255)
axs[0].set_title("logits")
axs[1].imshow(segment_img_pil, vmin=0, vmax=segment_img_original.max()*255)
axs[1].set_title("gt");

plt.imshow(abs(logit_output/255.0-segment_img_original)*100);
plt.title("Differences on raw fused logits (%)")
plt.colorbar();
