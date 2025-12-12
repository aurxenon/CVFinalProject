## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Evaluation script for SloMoBlur dataset from local folder
## Based on demo.py and Motion_Deblurring/test.py

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
from runpy import run_path
from skimage import img_as_ubyte
import cv2
from tqdm import tqdm
import argparse
import numpy as np
from glob import glob
from natsort import natsorted
import json
import time

parser = argparse.ArgumentParser(description='Process blurred images with Restormer')
parser.add_argument('--input_dir', default='./Dataset/SloMoBlur/0_9999/blurred', type=str, 
                    help='Directory containing input blurred images')
parser.add_argument('--output_dir', default='./Dataset/SloMoBlur/0_9999/deblurred_original_size', type=str, 
                    help='Directory for deblurred output images')
parser.add_argument('--weights', default=None, 
                    type=str, help='Path to model weights (overrides default task weights)')
parser.add_argument('--task', default='Single_Image_Defocus_Deblurring', type=str, 
                    help='Task type', choices=['Motion_Deblurring', 'Single_Image_Defocus_Deblurring'])
parser.add_argument('--tile', type=int, default=None, 
                    help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, 
                    help='Overlapping of different tiles')
parser.add_argument('--json_output', default='./results.json', type=str,
                    help='Path to save JSON results file (default: ./results.json)')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for processing (default: 1). For full-resolimages, use smaller batches to avoid OOM.')
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of workers for data loading (default: 4)')

args = parser.parse_args()

# Normalize paths - convert /Dataset to ./Dataset if it doesn't exist
def normalize_path(path):
    """Normalize path, converting absolute /Dataset to relative if needed"""
    if path.startswith('/Dataset') and not os.path.exists(path):
        # Try relative path instead
        relative_path = path[1:]  # Remove leading /
        if os.path.exists(relative_path):
            return relative_path
    return path

args.input_dir = normalize_path(args.input_dir)
args.output_dir = normalize_path(args.output_dir)

def load_img(filepath):
    """Load image from filepath and convert to RGB format"""
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def save_img(filepath, img):
    """Save image in RGB format"""
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

class ImageDataset(data.Dataset):
    """Dataset class for loading images"""
    def __init__(self, file_paths, img_multiple_of=8):
        self.file_paths = file_paths
        self.img_multiple_of = img_multiple_of
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Load image
        blurred_np = load_img(file_path)
        
        # Get original dimensions
        original_height, original_width = blurred_np.shape[0], blurred_np.shape[1]
        
        # Ensure image has 3 channels
        if blurred_np.ndim == 2:
            blurred_np = np.expand_dims(blurred_np, axis=2)
        if blurred_np.shape[2] == 1:
            blurred_np = np.repeat(blurred_np, 3, axis=2)
        elif blurred_np.shape[2] == 4:  # RGBA
            blurred_np = blurred_np[:, :, :3]
        
        # Convert to float and normalize
        blurred_np = blurred_np.astype(np.float32) / 255.0
        
        # Convert to tensor and permute to CHW format
        input_tensor = torch.from_numpy(blurred_np).float().permute(2, 0, 1)
        
        # Get dimensions
        height, width = input_tensor.shape[1], input_tensor.shape[2]
        
        # Pad the input if not multiple of 8
        H = ((height + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        W = ((width + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        padh = H - height if height % self.img_multiple_of != 0 else 0
        padw = W - width if width % self.img_multiple_of != 0 else 0
        input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')
        
        return {
            'image': input_tensor,
            'file_path': file_path,
            'original_height': original_height,
            'original_width': original_width,
            'padded_height': H,
            'padded_width': W,
            'filename': os.path.basename(file_path)
        }

def collate_fn(batch):
    """Custom collate function to handle variable image sizes"""
    # If batch_size is 1, just return the single item
    if len(batch) == 1:
        item = batch[0]
        return {
            'images': item['image'].unsqueeze(0),
            'file_paths': [item['file_path']],
            'original_heights': [item['original_height']],
            'original_widths': [item['original_width']],
            'padded_heights': [item['padded_height']],
            'padded_widths': [item['padded_width']],
            'filenames': [item['filename']]
        }
    
    # For batches, pad all images to the same size (max height and width in batch)
    max_h = max([item['image'].shape[1] for item in batch])
    max_w = max([item['image'].shape[2] for item in batch])
    
    # Ensure max dimensions are multiples of 8
    img_multiple_of = 8
    max_h = ((max_h + img_multiple_of) // img_multiple_of) * img_multiple_of
    max_w = ((max_w + img_multiple_of) // img_multiple_of) * img_multiple_of
    
    batched_images = []
    file_paths = []
    original_heights = []
    original_widths = []
    padded_heights = []
    padded_widths = []
    filenames = []
    
    for item in batch:
        img = item['image']
        h, w = img.shape[1], img.shape[2]
        
        # Pad to max size
        padh = max_h - h
        padw = max_w - w
        img_padded = F.pad(img, (0, padw, 0, padh), 'reflect')
        
        batched_images.append(img_padded)
        file_paths.append(item['file_path'])
        original_heights.append(item['original_height'])
        original_widths.append(item['original_width'])
        padded_heights.append(item['padded_height'])
        padded_widths.append(item['padded_width'])
        filenames.append(item['filename'])
    
    return {
        'images': torch.stack(batched_images),
        'file_paths': file_paths,
        'original_heights': original_heights,
        'original_widths': original_widths,
        'padded_heights': padded_heights,
        'padded_widths': padded_widths,
        'filenames': filenames
    }

def get_weights_and_parameters(task, parameters):
    """Get model weights path and update parameters based on task"""
    if task == 'Motion_Deblurring':
        weights = os.path.join('Motion_Deblurring', 'pretrained_models', 'motion_deblurring.pth')
    elif task == 'Single_Image_Defocus_Deblurring':
        weights = os.path.join('Defocus_Deblurring', 'pretrained_models', 'single_image_defocus_deblurring_fine_tuned.pth')
        # Note: Single_Image_Defocus_Deblurring uses 'WithBias' (default), not 'BiasFree'
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    # Override with user-specified weights if provided and exists
    if args.weights and os.path.exists(args.weights):
        weights = args.weights
    
    return weights, parameters

# Setup directories
task = args.task
input_dir = args.input_dir
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Get model weights and parameters
parameters = {
    'inp_channels': 3, 
    'out_channels': 3, 
    'dim': 48, 
    'num_blocks': [4, 6, 6, 8], 
    'num_refinement_blocks': 4, 
    'heads': [1, 2, 4, 8], 
    'ffn_expansion_factor': 2.66, 
    'bias': False, 
    'LayerNorm_type': 'WithBias', 
    'dual_pixel_task': False
}
weights, parameters = get_weights_and_parameters(task, parameters)

# Load model architecture
load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"===> Using device: {device}")
if torch.cuda.is_available():
    print(f"===> GPU: {torch.cuda.get_device_name(0)}")
    print(f"===> CUDA Version: {torch.version.cuda}")
    print(f"===> GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

model.to(device)

# Load weights (matching demo.py exactly)
checkpoint = torch.load(weights, map_location=device)
if 'params' in checkpoint:
    model.load_state_dict(checkpoint['params'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)
print(f"===> Loaded weights from: {weights}")
print(f"===> Model is on device: {next(model.parameters()).device}")

model.eval()

# Enable optimizations for faster inference
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed

img_multiple_of = 8

print(f"\n ==> Processing {task} on images from: {input_dir}")
print(f" ==> Using weights: {weights}")
print(f" ==> Results will be saved to: {output_dir}\n")

# Load image files from directory
if not os.path.exists(input_dir):
    # Try to suggest correct path
    if input_dir.startswith('/Dataset'):
        suggested_path = input_dir[1:]  # Remove leading /
        raise Exception(f'Input directory not found: {input_dir}\n'
                       f'Did you mean: {suggested_path} ?\n'
                       f'Or use relative path: ./Dataset/SloMoBlur/0_9999/blurred')
    else:
        raise Exception(f'Input directory not found: {input_dir}')

extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
files = []
for ext in extensions:
    files.extend(glob(os.path.join(input_dir, '*.' + ext)))
files = natsorted(files)

if len(files) == 0:
    raise Exception(f'No image files found at {input_dir}\n'
                   f'Please check that the directory contains image files (png, jpg, etc.)')

print(f"Found {len(files)} images to process")
print(f"Using batch size: {args.batch_size}, num_workers: {args.num_workers}")
print(f"Processing images at original resolution")

# Estimate memory requirements and adjust batch size
def estimate_memory_requirements(batch_size, img_h, img_w):
    """Estimate GPU memory needed for a batch"""
    # Input tensor memory (float32)
    input_mem = batch_size * 3 * img_h * img_w * 4 / (1024**3)  # GB
    
    # Model intermediate activations (roughly 5-10x input size for transformers)
    # Using conservative estimate of 8x
    activation_mem = input_mem * 8
    
    # Output tensor memory
    output_mem = input_mem
    
    # Total estimated memory
    total_mem = input_mem + activation_mem + output_mem
    
    return total_mem, input_mem, activation_mem, output_mem

def find_optimal_batch_size(max_batch_size, img_h, img_w, max_memory_gb=25.0):
    """Find optimal batch size that fits in GPU memory"""
    for bs in range(max_batch_size, 0, -1):
        total_mem, _, _, _ = estimate_memory_requirements(bs, img_h, img_w)
        if total_mem <= max_memory_gb:
            return bs, total_mem
    return 1, estimate_memory_requirements(1, img_h, img_w)[0]

# Verify GPU is being used and estimate memory
if torch.cuda.is_available():
    # Test GPU with a dummy tensor
    test_tensor = torch.randn(1, 3, 224, 224).to(device)
    _ = model(test_tensor)
    torch.cuda.synchronize()
    print(f"✓ GPU test successful - model is running on {device}")
    
    # Get GPU memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)  # GB
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)  # GB
    free_memory = total_memory - reserved_memory
    
    print(f"✓ GPU Memory: {total_memory:.2f} GB total, {allocated_memory:.2f} GB allocated, {reserved_memory:.2f} GB reserved")
    print(f"✓ Available GPU memory: {free_memory:.2f} GB")
    
    # Get sample image size to estimate memory
    if len(files) > 0:
        sample_img = load_img(files[0])
        sample_h, sample_w = sample_img.shape[0], sample_img.shape[1]
        # Account for padding to multiple of 8
        padded_h = ((sample_h + 7) // 8) * 8
        padded_w = ((sample_w + 7) // 8) * 8
        
        print(f"\n📊 Memory Estimation:")
        print(f"   Sample image size: {sample_h}x{sample_w} (padded to {padded_h}x{padded_w})")
        
        # Estimate for current batch size
        est_total, est_input, est_activation, est_output = estimate_memory_requirements(
            args.batch_size, padded_h, padded_w
        )
        print(f"   Estimated memory for batch_size={args.batch_size}:")
        print(f"     - Input: {est_input:.2f} GB")
        print(f"     - Activations: {est_activation:.2f} GB")
        print(f"     - Output: {est_output:.2f} GB")
        print(f"     - Total: {est_total:.2f} GB")
        
        # Check if it fits
        if est_total > free_memory:
            print(f"\n⚠ WARNING: Estimated memory ({est_total:.2f} GB) exceeds available ({free_memory:.2f} GB)")
            optimal_bs, optimal_mem = find_optimal_batch_size(args.batch_size, padded_h, padded_w, free_memory)
            print(f"   Recommended batch_size: {optimal_bs} (estimated memory: {optimal_mem:.2f} GB)")
            if optimal_bs < args.batch_size:
                print(f"   ⚠ Auto-adjusting batch_size from {args.batch_size} to {optimal_bs}")
                args.batch_size = optimal_bs
        else:
            print(f"   ✓ Memory estimate fits in available GPU memory")

# Create dataset and dataloader
dataset = ImageDataset(files, img_multiple_of=img_multiple_of)
dataloader = DataLoader(
    dataset, 
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True if torch.cuda.is_available() else False,
    collate_fn=collate_fn
)

# Storage for results
results = []

with torch.no_grad():
    gpu_time_total = 0
    cpu_time_total = 0
    num_batches = 0
    current_batch_size = args.batch_size
    oom_count = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing images")):
        try:
            # Clear cache before each batch to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move batch to GPU (non_blocking for async transfer)
            images = batch['images'].to(device, non_blocking=True)
            
            file_paths = batch['file_paths']
            original_heights = batch['original_heights']
            original_widths = batch['original_widths']
            padded_heights = batch['padded_heights']
            padded_widths = batch['padded_widths']
            filenames = batch['filenames']
            
            batch_size = images.shape[0]
            
            # Run inference
            if args.tile is None:
                # Testing on the original resolution image - process entire batch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure previous operations complete
                    gpu_start = time.time()
                
                restored = model(images)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Wait for GPU to finish
                    gpu_time_total += time.time() - gpu_start
                    oom_count = 0  # Reset OOM counter on success
            else:
                # Tile processing - process each image in batch individually
                restored_list = []
                for i in range(batch_size):
                    img = images[i:i+1]  # Keep batch dimension
                    b, c, h, w = img.shape
                    tile = min(args.tile, h, w)
                    assert tile % 8 == 0, "tile size should be multiple of 8"
                    tile_overlap = args.tile_overlap
                    
                    stride = tile - tile_overlap
                    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                    E = torch.zeros(b, c, h, w, device=device).type_as(img)
                    W = torch.zeros_like(E)
                    
                    for h_idx in h_idx_list:
                        for w_idx in w_idx_list:
                            in_patch = img[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                            out_patch = model(in_patch)
                            out_patch_mask = torch.ones_like(out_patch)
                            
                            E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                            W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
                    restored_list.append(E.div_(W))
                
                restored = torch.cat(restored_list, dim=0)
            
            restored = torch.clamp(restored, 0, 1)
            
            # Move entire batch to CPU at once (more efficient)
            cpu_start = time.time()
            restored_cpu = restored.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
            
            # Process each image in the batch
            for i in range(batch_size):
                try:
                    # Unpad the output to original size (remove padding)
                    original_h = original_heights[i]
                    original_w = original_widths[i]
                    restored_np = restored_cpu[i, :original_h, :original_w, :]
                    
                    # Ensure values are in [0, 1] range (handle floating point precision issues)
                    restored_np = np.clip(restored_np, 0.0, 1.0).astype(np.float32)
                    
                    # Convert to uint8 (0-255) - img_as_ubyte expects [0, 1] range for float images
                    try:
                        restored_uint8 = img_as_ubyte(restored_np)
                    except ValueError as ve:
                        # Fallback: manual conversion if img_as_ubyte fails
                        print(f"Warning: img_as_ubyte failed, using manual conversion: {ve}")
                        restored_uint8 = (np.clip(restored_np * 255.0, 0, 255)).astype(np.uint8)
                    
                    # Get output filename
                    input_filename = filenames[i]
                    output_filename = os.path.splitext(input_filename)[0] + '.png'
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Save deblurred image
                    save_img(output_path, restored_uint8)
                    
                    # Store results
                    results.append({
                        'input_file': file_paths[i],
                        'output_file': output_path,
                        'filename': input_filename
                    })
                    
                except Exception as e:
                    print(f"Error saving {file_paths[i]}: {e}")
                    continue
            
            cpu_time_total += time.time() - cpu_start
            num_batches += 1
        
        except torch.cuda.OutOfMemoryError as e:
            oom_count += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\n❌ CUDA Out of Memory at batch {batch_idx}")
            print(f"   Current batch_size: {current_batch_size}")
            print(f"   Error: {str(e)[:200]}")
            
            if current_batch_size > 1:
                # Try reducing batch size
                current_batch_size = max(1, current_batch_size // 2)
                print(f"   ⚠ Reducing batch_size to {current_batch_size}")
                print(f"   ⚠ Please restart the script with --batch_size {current_batch_size}")
                print(f"   ⚠ Or use --tile 512 to process images in tiles")
                # Skip this batch
                continue
            else:
                print(f"   ❌ Batch size is already 1. Cannot reduce further.")
                print(f"   💡 Solutions:")
                print(f"      1. Use tiling: --tile 512 or --tile 256")
                print(f"      2. Process images individually")
                print(f"      3. Use a GPU with more memory")
                # Skip this batch
                continue
                
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

print(f"\n{'='*60}")
print(f"Processing Complete")
print(f"{'='*60}")
print(f"Total images processed: {len(results)}/{len(files)}")
if torch.cuda.is_available():
    final_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
    final_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
    print(f"Final GPU memory: {final_allocated:.2f} GB allocated, {final_reserved:.2f} GB reserved")
if num_batches > 0 and torch.cuda.is_available():
    avg_gpu_time = gpu_time_total / num_batches * 1000  # ms
    avg_cpu_time = cpu_time_total / num_batches * 1000  # ms
    print(f"Average GPU inference time per batch: {avg_gpu_time:.2f} ms")
    print(f"Average CPU post-processing time per batch: {avg_cpu_time:.2f} ms")
    print(f"GPU utilization: {gpu_time_total / (gpu_time_total + cpu_time_total) * 100:.1f}% of total time")
    if avg_cpu_time > avg_gpu_time * 2:
        print(f"⚠ WARNING: CPU operations are {avg_cpu_time/avg_gpu_time:.1f}x slower than GPU!")
        print(f"   Consider using larger batch_size to keep GPU busy longer")
if oom_count > 0:
    print(f"⚠ Encountered {oom_count} out-of-memory errors")
    print(f"   Consider using --batch_size 1 or --tile 512")
print(f"{'='*60}")

# Save results to JSON
results_summary = {
    'task': task,
    'weights': weights,
    'input_dir': input_dir,
    'output_dir': output_dir,
    'total_images': len(results),
    'processed_files': results
}

json_output_path = args.json_output
with open(json_output_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults saved to:")
print(f"  - Deblurred images: {output_dir}")
print(f"  - Results JSON: {json_output_path}")

