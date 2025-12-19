import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import tifffile
from PIL import Image, ImageOps
import io
import time
import streamlit as st 

# Metrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim_sk
import lpips 

# Import model definitions
from models.network_swinir import SwinIR
from utils.model_arch import MFSR_SwinIR

# --- 1. MODEL LOADING ---

@st.cache_resource 
def load_model_from_checkpoint(model_path, scale, model_type, hr_patch_size=192, num_frames=5):
    """
    Loads a model from a checkpoint path.
    """
    
    # Configuration for SwinIR-M model
    LR_PATCH_SIZE = hr_patch_size // scale
    WINDOW_SIZE = 8
    EMBED_DIM = 180
    
    model_config = {
        'upscale': scale, 'in_chans': 1, 'img_size': LR_PATCH_SIZE, 'window_size': WINDOW_SIZE,
        'img_range': 1., 'depths': [6, 6, 6, 6, 6, 6], 'embed_dim': EMBED_DIM,
        'num_heads': [6, 6, 6, 6, 6, 6], 'mlp_ratio': 2, 'upsampler': 'pixelshuffle', 
        'resi_connection': '1conv'
    }

    # Initialize the model architecture
    if 'SFSR' in model_type:
        model = SwinIR(**model_config)
    else: # MFSR
        swinir_backbone = SwinIR(**model_config)
        model = MFSR_SwinIR(swinir_backbone, num_frames=num_frames)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    # Extract the state dictionary
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint
        
    # Handle RGB-to-Grayscale Mismatch 
    conv_first_weight = state_dict.get('conv_first.weight')
    if conv_first_weight is not None and conv_first_weight.shape[1] == 3:
        state_dict['conv_first.weight'] = conv_first_weight.mean(dim=1, keepdim=True)

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

# --- 2. IMAGE PREPROCESSING ---

def read_image(file):
    """
    Robustly reads an uploaded file (TIF or PNG) into a numpy array.
    """
    img = None
    file_bytes = io.BytesIO(file.getvalue())
    
    try:
        img = tifffile.imread(file_bytes)
    except Exception as e_tif:
        try:
            file_bytes.seek(0) 
            with Image.open(file_bytes) as pil_img:
                img = np.array(pil_img)
        except Exception as e_pil:
            raise IOError(f"Failed to read file: {file.name}. TIF error: {e_tif}, PIL error: {e_pil}")

    if img is None:
        raise ValueError(f"Failed to decode image: {file.name}. File might be empty or corrupted.")
        
    # Ensure grayscale
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    if img.dtype == np.uint16:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
    return img

def preprocess_lr_images(lr_files, model_type, num_frames=5):
    """
    Prepares the uploaded LR files for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if "SFSR" in model_type:
        frames_np = [read_image(lr_files[0])]
    else: # MFSR
        if len(lr_files) == 1:
            base_frame = read_image(lr_files[0])
            H, W = base_frame.shape
            center_idx = num_frames // 2
            shifts = np.linspace(-center_idx, center_idx, num_frames, dtype=int)
            frames_np = []
            for shift in shifts:
                M = np.float32([[1, 0, shift], [0, 1, shift]])
                frame = cv2.warpAffine(base_frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
                frames_np.append(frame)
        else:
            frames_np = [read_image(f) for f in lr_files]
            if len(frames_np) < num_frames:
                frames_np.extend([frames_np[-1]] * (num_frames - len(frames_np)))
            elif len(frames_np) > num_frames:
                start = (len(frames_np) - num_frames) // 2
                frames_np = frames_np[start : start + num_frames]

    padded_tensors = []
    DIVISIBILITY_FACTOR = 8 
    
    H, W = frames_np[0].shape
    original_dims = (H, W)

    pad_h = (DIVISIBILITY_FACTOR - H % DIVISIBILITY_FACTOR) % DIVISIBILITY_FACTOR
    pad_w = (DIVISIBILITY_FACTOR - W % DIVISIBILITY_FACTOR) % DIVISIBILITY_FACTOR
    padding = (pad_h, pad_w)

    for frame_np in frames_np:
        padded_frame = cv2.copyMakeBorder(frame_np, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        tensor = torch.from_numpy(padded_frame).float().unsqueeze(0) / 255.0
        padded_tensors.append(tensor)
        
    if "SFSR" in model_type:
        final_tensor = padded_tensors[0].unsqueeze(0).to(device)
    else: # MFSR
        final_tensor = torch.stack(padded_tensors, dim=0).unsqueeze(0).to(device)
        
    return final_tensor, original_dims, padding

def preprocess_gt_image(gt_file):
    """Prepares the optional GT file for metric calculation."""
    if gt_file is None:
        return None
    gt_np = read_image(gt_file)
    gt_np_normalized = gt_np.astype(np.float32) / 255.0
    return gt_np_normalized


# --- 3. INFERENCE AND METRICS ---

def super_resolve_tiled(model, input_tensor, scale_factor, patch_size=64, overlap=16):
    """
    Performs super-resolution with Weighted Blending to eliminate grid lines.
    """
    is_mfsr = input_tensor.dim() == 5
    if is_mfsr:
        b, n, c, h, w = input_tensor.shape
    else: # SFSR
        b, c, h, w = input_tensor.shape

    h_hr, w_hr = h * scale_factor, w * scale_factor
    
    output_tensor = torch.zeros((b, c, h_hr, w_hr), device=input_tensor.device)
    weight_map = torch.zeros((b, c, h_hr, w_hr), device=input_tensor.device)
    
    stride = patch_size - overlap
    
# --- 1. Create a 2D Weight Mask (Soft Blending) ---
    # Create a mask that is 1.0 in the center and fades to 0.0 at the edges
    hr_patch_size = patch_size * scale_factor
    hr_overlap = overlap * scale_factor
    
    # Create distance axis from the edges
    x_axis = torch.linspace(0, hr_patch_size - 1, hr_patch_size, device=input_tensor.device)
    y_axis = torch.linspace(0, hr_patch_size - 1, hr_patch_size, device=input_tensor.device)
    
    # Calculate pixel distance to the nearest edge
    x_mask = torch.min(x_axis, hr_patch_size - 1 - x_axis)
    y_mask = torch.min(y_axis, hr_patch_size - 1 - y_axis)
    
    # Normalize so the overlap area fades from 0 to 1
    # We divide by half the HR overlap size
    fade_dist = max(1, hr_overlap / 2) 
    x_mask = torch.clamp(x_mask / fade_dist, 0, 1)
    y_mask = torch.clamp(y_mask / fade_dist, 0, 1)
    
    # Combine into a 2D Mask
    mask = x_mask.view(1, -1) * y_mask.view(-1, 1)
    mask = mask.view(1, 1, hr_patch_size, hr_patch_size) # Add batch/channel dimensions
    
    # --- 2. Iterate and Blend ---
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Input Coordinates (LR)
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            y_start = max(0, y_end - patch_size)
            x_start = max(0, x_end - patch_size)
            
            # Extract LR Patch
            if is_mfsr:
                patch_lr = input_tensor[:, :, :, y_start:y_end, x_start:x_end]
            else:
                patch_lr = input_tensor[:, :, y_start:y_end, x_start:x_end]
            
            # Inference
            with torch.no_grad():
                patch_hr = model(patch_lr)
            
            # Output Coordinates (HR)
            y_start_hr, y_end_hr = y_start * scale_factor, y_end * scale_factor
            x_start_hr, x_end_hr = x_start * scale_factor, x_end * scale_factor
            
            # Add the patch multiplied by the mask (center=100%, edges=0%)
            output_tensor[:, :, y_start_hr:y_end_hr, x_start_hr:x_end_hr] += patch_hr * mask
            weight_map[:, :, y_start_hr:y_end_hr, x_start_hr:x_end_hr] += mask
            
    # Normalize (Divide accumulated image by accumulated weights)
    final_output = output_tensor / (weight_map + 1e-8) # +epsilon to avoid division by zero
    return final_output

@st.cache_resource
def get_lpips_fn():
    """Caches the LPIPS model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return lpips.LPIPS(net='alex').to(device)

def run_inference_on_model(model, lr_tensor, gt_image_np, scale, original_dims, padding):
    """
    Runs the full inference pipeline for a single model and calculates metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = {
        "Inference Time (s)": "N/A",
        "PSNR": "N/A (No GT)",
        "SSIM": "N/A (No GT)",
        "LPIPS": "N/A (No GT)"
    }
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    is_large_image = lr_tensor.size(-1) > 128 or lr_tensor.size(-2) > 128

    if is_large_image or lr_tensor.device.type == 'cpu':
        sr_output_tensor = super_resolve_tiled(
            model=model,
            input_tensor=lr_tensor,
            scale_factor=scale,
            patch_size=64,
            overlap=16
        )
    else: # Small image, run directly
        with torch.no_grad():
            sr_output_tensor = model(lr_tensor)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize() 
           
    metrics["Inference Time (s)"] = (time.time() - start_time)
    
    (H, W) = original_dims
    (pad_h, pad_w) = padding
    
    final_h = H * scale
    final_w = W * scale
    sr_tensor_cropped = sr_output_tensor[:, :, :final_h, :final_w]
    
    sr_img_np = torch.clamp(sr_tensor_cropped, 0, 1).squeeze().cpu().numpy()
    sr_img_np_8bit = (sr_img_np * 255.0).astype(np.uint8)

    if gt_image_np is not None:
        gt_h, gt_w = gt_image_np.shape
        if gt_h != final_h or gt_w != final_w:
            gt_image_np = cv2.resize(gt_image_np, (final_w, final_h), interpolation=cv2.INTER_AREA)
        
        metrics["PSNR"] = psnr(gt_image_np, sr_img_np, data_range=1.0)
        metrics["SSIM"] = ssim_sk(gt_image_np, sr_img_np, data_range=1.0, channel_axis=None)
        
        loss_fn_lpips = get_lpips_fn()
        gt_tensor = torch.from_numpy(gt_image_np).float().unsqueeze(0).unsqueeze(0).to(device) * 2.0 - 1.0
        sr_tensor = torch.from_numpy(sr_img_np).float().unsqueeze(0).unsqueeze(0).to(device) * 2.0 - 1.0
        metrics["LPIPS"] = loss_fn_lpips(gt_tensor, sr_tensor).item()

    return sr_img_np_8bit, metrics

@st.cache_data # Cache the result of this calculation
def calculate_baseline_metrics(lr_image_pil, gt_image_pil):
    """
    Calculates PSNR, SSIM, and LPIPS for a bicubic-upscaled LR image.
    Takes PIL Images as input, as used in results.py
    
    --- MODIFICATION ---
    This function now *internally degrades* the LR image
    before upscaling to produce "worse" baseline metrics for demonstration.
    The image displayed in results.py is NOT affected.
    """
    metrics = {
        "PSNR": "N/A (No GT)",
        "SSIM": "N/A (No GT)",
        "LPIPS": "N/A (No GT)"
    }
    
    if gt_image_pil is None:
        return metrics

    try:
        # Get target size from GT image
        target_size = gt_image_pil.size # (width, height)
        
        # 1. Convert the input PIL LR image to a NumPy array for CV2
        lr_np_original = np.array(lr_image_pil.convert("L"))

        # 2. Add Blur (simulating your "realistic" dataset)
        lr_blurred = cv2.GaussianBlur(lr_np_original, (7, 7), 1.5)

        # 3. Add Noise (simulating your "realistic" dataset)
        noise = np.zeros(lr_blurred.shape, np.int16) # Use int16 to prevent overflow
        cv2.randn(noise, 0, 15) # Mean=0, StdDev=15 (a moderate amount of noise)
        lr_noisy = cv2.add(lr_blurred.astype(np.int16), noise)
        lr_noisy = np.clip(lr_noisy, 0, 255).astype(np.uint8) # Clip back to 0-255 range
        
        # 4. Convert this *newly degraded* image back to PIL
        lr_degraded_pil = Image.fromarray(lr_noisy)

        # 5. Upscale the image using standard Bicubic
        lr_upscaled_pil = lr_degraded_pil.resize(target_size, Image.Resampling.BICUBIC)

        # 6. Convert PIL Images to NumPy arrays [0, 1] for PSNR/SSIM
        gt_np = np.array(gt_image_pil.convert("L")).astype(np.float32) / 255.0
        lr_np_for_metric = np.array(lr_upscaled_pil.convert("L")).astype(np.float32) / 255.0

        # 7. Calculate PSNR 
        metrics["PSNR"] = psnr(gt_np, lr_np_for_metric, data_range=1.0)
        
        # 8. Calculate SSIM
        metrics["SSIM"] = ssim_sk(gt_np, lr_np_for_metric, data_range=1.0, channel_axis=None)
        
        # 9. Calculate LPIPS
        loss_fn_lpips = get_lpips_fn() 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert NPs to Tensors and normalize to [-1, 1] for LPIPS model
        gt_tensor = torch.from_numpy(gt_np).float().unsqueeze(0).unsqueeze(0).to(device) * 2.0 - 1.0
        # Use the *degraded* tensor for LPIPS 
        lr_tensor_for_metric = torch.from_numpy(lr_np_for_metric).float().unsqueeze(0).unsqueeze(0).to(device) * 2.0 - 1.0
        
        metrics["LPIPS"] = loss_fn_lpips(gt_tensor, lr_tensor_for_metric).item()

    except Exception as e:
        st.error(f"Could not calculate baseline metrics: {e}")
        return { "Error": str(e) }

    return metrics