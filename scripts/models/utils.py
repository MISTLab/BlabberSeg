import numpy as np
import torch

# https://github.com/MISTLab/DOVESEI/blob/fbc035a4e0fa5c3490703a8e91162e16714ea5aa/src/ros2_open_voc_landing_heatmap/ros2_open_voc_landing_heatmap/generate_landing_heatmap.py#L115
def convert2mask(logits, safety_threshold = .8, seg_dynamic_threshold = .1, dynamic_threshold_maxsteps = 100):
    logits_threshold = logits / 255 > safety_threshold
    curr_threshold = safety_threshold
    if seg_dynamic_threshold > 0:
        total_pixels = np.prod(logits.shape)
        threshold_step = safety_threshold/dynamic_threshold_maxsteps
        for ti in range(0, dynamic_threshold_maxsteps + 1):
            if (logits_threshold == True).sum() / total_pixels < seg_dynamic_threshold:
                curr_threshold = (safety_threshold - threshold_step * ti)
                logits_threshold = logits / 255 > curr_threshold
            else:
                break
    
    return logits_threshold.astype('uint8')

# Based on:
# - https://github.com/pytorch/vision/blob/9d0a93eee90bf7c401b74ebf9c8be80346254f15/references/segmentation/utils.py#L66
class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype = torch.int64, device = a.device)
        with torch.inference_mode():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength = n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return ("global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}").format(
            acc_global.item() * 100,
            [f"{i:.1f}" for i in (acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            iu.mean().item() * 100,
        )
