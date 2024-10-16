# For some reason the TensorRT engines are not reused in some situations, even when the model is exactly the same,  therefore they are re-created and that takes a long time.     

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

import platform
import pickle
from glob import glob
import shutil
import os

from PIL import Image
import numpy as np
import cv2
from timeit import default_timer as timer
from models.clipseg_mod import (CLIPActivations, CLIPSegDecoder, CLIPSegDecoderProcessConditional) 
import torch
from torchvision import transforms
import torch.onnx
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device = {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))
print(platform.uname())

# Values according to DOVESEI's defaults
img_size = 352
safety_threshold = .8 
blur_kernel_size = 15
seg_dynamic_threshold = .1
DYNAMIC_THRESHOLD_MAXSTEPS = 100
len_positive_prompts = 3

print(f"Image size: {img_size, img_size}")
print(f"Safety_threshold: {safety_threshold}")
print(f"Blur kernel size: {blur_kernel_size}")
print(f"Segmentation dynamic threshold: {seg_dynamic_threshold}")
print(f"Segmentation dynamic threshold max steps: {DYNAMIC_THRESHOLD_MAXSTEPS}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225]), # std. ImageNet stats.
    transforms.Resize((img_size, img_size), antialias = True),
])

# As far as I understood, the best way is to use base ONNX models with float32 precision as the following conversions seem to work better that way. On the other hand, a model that is float16 and receives float16 should spend less time moving inputs and outputs between CPU and GPU.

# This model has the transforms.Normalize(mean = [.485, .456, .406], std = [.229, .224, .225]) inside the model to avoid slow downs when torch is not available (e.g. Jetson Nano 2GB).

# FP16 
activations_model = CLIPActivations(img_size = img_size)
activations_model.load_state_dict(torch.load('CLIPActivations/CLIP_float16.pth', map_location = torch.device(device)), strict = False)
activations_model.load_state_dict(torch.load(f'CLIPActivations/CLIPActivations_float16_{img_size}.pth', map_location = torch.device(device)), strict = False)
activations_model.eval()
activations_model.cuda().half()

seg_model = CLIPSegDecoder(img_size = img_size, batch_size = 1)
seg_model.load_state_dict(torch.load('weights/rd64-uni-refined.pth', map_location = torch.device(device)), strict = False)
seg_model.eval()
seg_model.cuda().half()

fake_input = torch.rand(1, 3, 352, 352, dtype = torch.float16, device = 'cuda')
fake_cond = torch.rand(1, 64).cuda().half()
fake_activs = torch.rand(3, 485, 1, 768).cuda().half()

# Export the model
torch.onnx.export(activations_model.cuda(),                  # Model to be converted
                  fake_input,                                # Model input (or a tuple for multiple inputs)
                  "onnx/CLIPActivations_fp16_352.onnx",      # Where to save the model (can be a file or file-like object)
                  export_params = True,                      # Store the trained parameter weights inside the model file
                  opset_version = 14,                        # ONNX version to export the model
                  do_constant_folding = True,                # Whether to execute constant folding for optimization
                  input_names = ['input_tensor'],            # Model's input names
                  output_names = ['activations3', 
                                  'activations6', 
                                  'activations9'],           # Model's output names
                  dynamic_axes = None,
                  verbose = True
                  )

torch.onnx.export(seg_model.cuda(),                         # Model to be converted
                  (fake_cond, fake_cond, *fake_activs),     # Model input (or a tuple for multiple inputs)
                  "onnx/CLIPSegDecoder_fp16_352.onnx",      # Where to save the model (can be a file or file-like object)
                  export_params = True,                     # Store the trained parameter weights inside the model file
                  opset_version = 14,                       # ONNX version to export the model
                  do_constant_folding = True,               # Whether to execute constant folding for optimization
                  input_names = ['cond0', 'cond1', 
                                 'activations3',
                                 'activations6',
                                 'activations9'],           # Model's input names
                  output_names = ['segmentation'],          # Model's output names
                  dynamic_axes = None
                  )
