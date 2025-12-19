import os
import random
import cv2
import numpy as np
import tifffile
from pathlib import Path
from skimage.util import random_noise
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = "Dataset/Split Dataset"
SCALE_FACTOR = 4   # 4 for x4 scaling, 2 for x2
NUM_FRAMES = 5     # Images to generate per ground truth
SHIFT_LIMIT = 0.5  # Max subpixel shift

def apply_shift(img, x, y):
    """Applies subpixel shift using affine transformation."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def generate_lr_dataset(root_path):
    root = Path(root_path)
    
    # Recursively find all TIFF files in 'ground_truth' folders
    files = list(root.rglob("ground_truth/*.tif*"))
    
    if not files:
        print(f"No images found in {root}")
        return

    print(f"Processing {len(files)} images...")

    for hr_path in tqdm(files):
        try:
            # Setup paths
            gt_dir = hr_path.parent
            base_dir = gt_dir.parent
            stem = hr_path.stem 
            
            # Create output directories
            bicubic_dir = base_dir / 'lr_bicubic'
            realistic_dir = base_dir / 'lr_realistic'
            bicubic_dir.mkdir(exist_ok=True)
            realistic_dir.mkdir(exist_ok=True)

            # Load and Normalize Image
            hr_img = tifffile.imread(str(hr_path))
            if hr_img.dtype != np.uint8:
                hr_img = cv2.normalize(hr_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            h, w = hr_img.shape
            lr_h, lr_w = h // SCALE_FACTOR, w // SCALE_FACTOR

            for i in range(NUM_FRAMES):
                # 1. Random Subpixel Shift
                sx = random.uniform(-SHIFT_LIMIT, SHIFT_LIMIT)
                sy = random.uniform(-SHIFT_LIMIT, SHIFT_LIMIT)
                shifted = apply_shift(hr_img, sx, sy)

                # 2. Generate 'Bicubic' (Clean)
                bicubic = cv2.resize(shifted, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
                
                # 3. Generate 'Realistic' (Blur -> Noise -> Resize)
                # Randomize blur kernel and sigma
                k_size = random.choice(range(7, 22, 2)) # Odds between 7 and 21
                sigma = random.uniform(0.2, 3.0)
                blurred = cv2.GaussianBlur(shifted, (k_size, k_size), sigma)
                
                # Add Gaussian + Poisson noise
                img_float = blurred.astype(np.float32) / 255.0
                var = random.uniform(0.0001, 0.005)
                noisy = random_noise(img_float, mode='gaussian', var=var, clip=True)
                noisy = random_noise(noisy, mode='poisson', clip=True)
                
                # Convert back to uint8 and resize
                degraded = (noisy * 255).astype(np.uint8)
                realistic = cv2.resize(degraded, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

                # Save files
                fname = f"{stem}_{i+1:02d}.png"
                cv2.imwrite(str(bicubic_dir / fname), bicubic)
                cv2.imwrite(str(realistic_dir / fname), realistic)

        except Exception as e:
            print(f"Error processing {hr_path.name}: {e}")

if __name__ == "__main__":
    generate_lr_dataset(SOURCE_DIR)