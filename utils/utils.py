import matplotlib.pyplot as plt


import torch
from torch import Tensor
from torchvision import transforms as T
from torch import nn
import random
import numpy as np
from typing import Literal, Optional, Tuple, Union
from datetime import datetime
from collections import deque
import json
import os
import time
import copy


def scale_stats_values(stats_dict, factor):
    """
    Multiplies all numerical values within nested 'min' and 'max' lists in the dictionary by a fixed factor.

    This function iterates through the top-level keys of the dictionary. If a key corresponds to another dictionary,
    and this dictionary contains keys named 'min' and 'max' (whose values are lists of numbers),
    then all elements within these two lists are multiplied by the specified 'factor'.

    Args:
        stats_dict (dict): The original dictionary containing statistical data.
                           Structure should be similar to {'qpos': {'min': [...], 'max': [...]}, ...}.
        factor (float or int): The multiplier used to scale the numerical values.

    Returns:
        dict: A new dictionary where 'min' and 'max' values have been scaled. The original dictionary is unaffected.
    """

    modified_dict = copy.deepcopy(stats_dict)

    for key, value in modified_dict.items():
        if isinstance(value, dict) and 'min' in value and 'max' in value:
            
            value['min'] = [num * factor for num in value['min']]
            
            value['max'] = [num * factor for num in value['max']]
            
    return modified_dict

def make_loss_recorder(log_step_freq: int,
                       save_dir: str = "./log/loss"):
    """
    record(loss, epoch, step)。
    """
    os.makedirs(save_dir, exist_ok=True)          
    loss_window = deque(maxlen=log_step_freq)     
    
    timestamp_start = datetime.now().strftime("%Y-%m-%d-%H-%M")
    file_path = os.path.join(save_dir, f"{timestamp_start}.json")
    
    record_dict = {
        "epoch":     [],
        "step":      [],
        "avg_loss":  []
    }
    
    def record(loss_val: float, epoch: int, step: int):
        loss_window.append(loss_val)

        if (step + 1) % log_step_freq != 0:
            return

        avg_loss  = sum(loss_window) / len(loss_window)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        
        record_dict['epoch'].append(epoch)
        record_dict['step'].append(step)
        record_dict['avg_loss'].append(avg_loss)

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(record_dict, fp, indent=4)

        print(f"[LOG] {timestamp} | epoch:{epoch} step:{step} "
              f"| avg_loss({log_step_freq} steps): {avg_loss:.6f}")

        loss_window.clear()       

    return record





class ActionSmoother:
    """
    A reusable class for performing 1D smoothing on (N, D) time-series actions.

    Parameters
    ----------
    window_size : int, default 5
        Must be a positive odd number. The length of the convolution window / moving average window.
    mode : {'mean', 'gauss'}, default 'mean'
        'mean'  – Equal-weight moving average
        'gauss' – Gaussian-weighted moving average
    sigma : float | None, default None
        Effective only when mode='gauss'. If None, defaults to window_size / 2.
    """

    def __init__(self,
                 window_size: int = 5,
                 mode: Literal['mean', 'gauss'] = 'mean',
                 sigma: Optional[float] = None):

        if window_size % 2 == 0 or window_size < 1:
            raise ValueError("window_size should > 0")
        if mode not in ('mean', 'gauss'):
            raise ValueError("mode can only be 'mean' or 'gauss'")

        self.window_size = window_size
        self.mode = mode

        if mode == 'mean':
            kernel = np.ones(window_size, dtype=np.float32) / window_size
        else:                        # Gaussian
            if sigma is None:
                sigma = window_size / 2.0
            self.sigma = sigma
            idx = np.arange(window_size) - window_size // 2
            kernel = np.exp(-0.5 * (idx / sigma) ** 2).astype(np.float32)
            kernel /= kernel.sum()

        self.kernel = kernel[:, None]

    def smooth(self, actions: np.ndarray) -> np.ndarray:
 
        if actions.ndim != 2:
            raise ValueError("actions must be a 2D array (N, D)")
        N, D = actions.shape

        pad = self.window_size // 2
        padded = np.pad(actions, ((pad, pad), (0, 0)), mode='edge')

        out = np.empty_like(actions, dtype=np.float32)
        for t in range(N):
            window = padded[t : t + self.window_size]       # (window, D)
            out[t] = np.sum(window * self.kernel, axis=0)
        return out

    __call__ = smooth

    def __repr__(self):
        base = f"ActionSmoother(window_size={self.window_size}, mode='{self.mode}'"
        if self.mode == 'gauss':
            base += f", sigma={getattr(self, 'sigma', None):.3g}"
        return base + ")"
    
    
    
class AddGaussianNoise:
    """
    Adds Gaussian noise to an image tensor (B, C, H, W).
    The noise is added to pixel values in the 0-255 range.
    
    Parameters:
    ----------
    mean : float
        Mean of the noise distribution.
    std : float
        Standard deviation of the noise distribution.
    """
    def __init__(self, mean=0.0, std=10.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.std + self.mean
        
        noisy_tensor = tensor + noise
        
        noisy_tensor = torch.clamp(noisy_tensor, 0, 255)
        
        return noisy_tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
    
    
class CenterCropWithJitter:
    """
    Performs a center crop on an image tensor (B, C, H, W) and adds random positional jitter.
    
    Parameters:
    ----------
    crop_size : tuple[int, int]
        Target size after cropping (height, width).
    jitter_max : tuple[int, int]
        Maximum jitter range in the (height, width) directions.
        For example, (10, 20) means the jitter range in the height direction is [-10, 10], 
        and in the width direction is [-20, 20].
    """
    def __init__(self, crop_size, jitter_max=(0, 0)):
        self.crop_size = crop_size
        self.jitter_max = jitter_max
        # 确保 crop_size 和 jitter_max 是元组
        if isinstance(self.crop_size, int):
            self.crop_size = (self.crop_size, self.crop_size)
        if isinstance(self.jitter_max, int):
            self.jitter_max = (self.jitter_max, self.jitter_max)

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:

        _, _, h, w = img_tensor.shape
        crop_h, crop_w = self.crop_size

        if crop_h > h or crop_w > w:
            raise ValueError(f"Crop size {self.crop_size} is larger than input image size {(h, w)}")

        center_top = (h - crop_h) // 2
        center_left = (w - crop_w) // 2
        
        h_jitter = random.randint(-self.jitter_max[0], self.jitter_max[0])
        w_jitter = random.randint(-self.jitter_max[1], self.jitter_max[1])
        
        jittered_top = center_top + h_jitter
        jittered_left = center_left + w_jitter

        final_top = max(0, min(jittered_top, h - crop_h))
        final_left = max(0, min(jittered_left, w - crop_w))
        
        cropped_img = img_tensor[:, :, final_top:final_top + crop_h, final_left:final_left + crop_w]
        
        return cropped_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crop_size={self.crop_size}, jitter_max={self.jitter_max})"

    
    
def build_img_preprocess(
        crop_size=(480, 460),            
        crop_jitter_max=(0, 0),          
        resize_size=(224, 224),
        brightness_contrast=None,        
        noise_std=None,                  
        train_flag=True
        ):                                
    """
    Generates an image preprocessing pipeline based on the provided crop and resize dimensions.
    
    Parameters
    ----
    crop_size   tuple[int,int] | None   Center crop dimensions. None indicates skipping this step.
    resize_size tuple[int,int] | None   Resize dimensions. None indicates skipping this step.
    """
    transforms = [lambda x: x.to(torch.float)]        
    
    if train_flag:  # train
        if crop_size is not None:
            transforms.append(CenterCropWithJitter(crop_size, crop_jitter_max))
    else:   # inference
        if crop_size is not None:
            transforms.append(T.CenterCrop(crop_size))   # (h, w)

    if resize_size is not None:
        transforms.append(T.Resize(resize_size))     # (h, w)
        
    if train_flag:
        if noise_std is not None and noise_std > 0:
            transforms.append(AddGaussianNoise(mean=0.0, std=noise_std))
            
    transforms.append(lambda x: x / 255.0)
    
    if train_flag:
        if brightness_contrast is not None and (brightness_contrast[0] > 0 or brightness_contrast[1] > 0):
            transforms.append(T.ColorJitter(brightness=brightness_contrast[0], contrast=brightness_contrast[1]))
        
    return T.Compose(transforms)



def preprocess_5d_tensor(x: torch.Tensor, transform_pipeline: T.Compose) -> torch.Tensor:
    """
    Applies standard torchvision transforms to a 5D tensor (B, N, C, H, W).
    
    Args:
        x: Input 5D tensor.
        transform_pipeline: A T.Compose object containing the transforms to apply.
        
    Returns:
        The transformed 5D tensor.
    """
    b, n = x.shape[0], x.shape[1]
    
    x_reshaped = x.view(b * n, *x.shape[2:])
    
    processed_reshaped = transform_pipeline(x_reshaped)
    
    output = processed_reshaped.view(b, n, *processed_reshaped.shape[1:])
    
    return output


def count_model_params(model):  
    total_params = sum(p.numel() for p in model.parameters())  
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    
    params_m = total_params / 1_000_000  
    params_b = total_params / 1_000_000_000  
    
    trainable_params_m = trainable_params / 1_000_000  
    trainable_params_b = trainable_params / 1_000_000_000   
    
    print(f"Parameters (M): {params_m:.2f}M")  
    print(f"Parameters (B): {params_b:.2f}B")  
    
    print(f"Trainable Params (M): {trainable_params_m:.2f}M")  
    print(f"Trainable Params (B): {trainable_params_b:.2f}B")  
    
    return {  
        "total_params": total_params,  
        "trainable_params": trainable_params,  
        "params_m": params_m,  
        "params_b": params_b  
    }  


def exp_decay_norm(x, N=5, k=0.8, w=1.):

    x = np.asarray(x, dtype=float)
    a = np.exp(w * k * x)
    b = np.exp(w * k * N)
    return (a - b) / (1 - b)




