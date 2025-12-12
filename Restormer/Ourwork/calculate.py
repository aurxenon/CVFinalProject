"""
Calculate PSNR and SSIM metrics between ground truth and deblurred images.
"""

import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
import json
from tqdm import tqdm
import torch

# Standalone PSNR and SSIM calculation functions to avoid lmdb dependency
def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order."""
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def bgr2ycbcr(img, y_only=False):
    """Convert BGR image to YCbCr color space.
    
    Args:
        img (ndarray): Image with range [0, 255] in BGR order.
        y_only (bool): Return only Y channel if True.
    
    Returns:
        ndarray: Image in YCbCr color space.
    """
    img = img.astype(np.float32) / 255.
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img,
            [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]
        ) + [16, 128, 128]
    out_img = out_img * 255.0
    out_img = np.clip(out_img, 0, 255).astype(np.uint8)
    return out_img


def to_y_channel(img):
    """Change to Y channel of YCbCr."""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.


def calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)."""
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    max_value = 1. if img1.max() <= 1 else 255.
    return 20. * np.log10(max_value / np.sqrt(mse))


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images."""
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _ssim_3d(img1, img2, max_value):
    """Calculate SSIM using 3D Gaussian kernel (GPU version)."""
    assert len(img1.shape) == 3 and len(img2.shape) == 3
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if torch.cuda.is_available():
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        kernel_3 = cv2.getGaussianKernel(11, 1.5)
        kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0)).float()
        
        # Create conv3d and move to GPU
        conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
        conv3d.weight.requires_grad = False
        conv3d.weight[0, 0, :, :, :] = kernel
        conv3d = conv3d.cuda()  # Move entire conv3d to GPU
        
        img1_t = torch.tensor(img1).float().cuda()
        img2_t = torch.tensor(img2).float().cuda()
        
        mu1 = conv3d(img1_t.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        mu2 = conv3d(img2_t.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = conv3d(img1_t.unsqueeze(0).unsqueeze(0) ** 2).squeeze(0).squeeze(0) - mu1_sq
        sigma2_sq = conv3d(img2_t.unsqueeze(0).unsqueeze(0) ** 2).squeeze(0).squeeze(0) - mu2_sq
        sigma12 = conv3d((img1_t * img2_t).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean().cpu().numpy())
    else:
        # Fallback to per-channel SSIM if no GPU
        ssims = []
        for i in range(img1.shape[2]):
            ssims.append(_ssim(img1[..., i], img2[..., i]))
        return np.array(ssims).mean()


def calculate_ssim(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):
    """Calculate SSIM (structural similarity)."""
    assert img1.shape == img2.shape, f'Image shapes are different: {img1.shape}, {img2.shape}.'
    
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)

    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)
        return _ssim(img1[..., 0], img2[..., 0])

    max_value = 1 if img1.max() <= 1 else 255
    return _ssim_3d(img1, img2, max_value)


def load_img(filepath):
    """Load image from filepath and convert to RGB format"""
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def calculate_metrics(gt_dir, deblurred_dir, output_json='metrics_results.json', crop_border=0, test_y_channel=False):
    """
    Calculate PSNR and SSIM metrics between ground truth and deblurred images.
    
    Args:
        gt_dir (str): Directory containing ground truth images
        deblurred_dir (str): Directory containing deblurred images
        output_json (str): Path to save results JSON file
        crop_border (int): Cropped pixels in each edge (default: 0)
        test_y_channel (bool): Test on Y channel of YCbCr (default: False)
    
    Returns:
        dict: Dictionary containing all results and statistics
    """
    
    # Get all image files from both directories
    extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
    gt_files = []
    for ext in extensions:
        gt_files.extend(glob(os.path.join(gt_dir, '*.' + ext)))
    gt_files = natsorted(gt_files)
    
    deblurred_files = []
    for ext in extensions:
        deblurred_files.extend(glob(os.path.join(deblurred_dir, '*.' + ext)))
    deblurred_files = natsorted(deblurred_files)
    
    print(f"Found {len(gt_files)} ground truth images")
    print(f"Found {len(deblurred_files)} deblurred images")
    
    # Match files by filename
    results = []
    psnr_values = []
    ssim_values = []
    
    # Create a dictionary of deblurred files by basename for faster lookup
    deblurred_dict = {}
    for deblurred_file in deblurred_files:
        basename = os.path.basename(deblurred_file)
        deblurred_dict[basename] = deblurred_file
    
    # Process each ground truth image
    for gt_file in tqdm(gt_files, desc="Calculating metrics"):
        gt_basename = os.path.basename(gt_file)
        
        # Find matching deblurred file
        if gt_basename not in deblurred_dict:
            # Try without extension or with different extension
            gt_name_no_ext = os.path.splitext(gt_basename)[0]
            found = False
            for deblurred_basename, deblurred_file in deblurred_dict.items():
                if os.path.splitext(deblurred_basename)[0] == gt_name_no_ext:
                    deblurred_file = deblurred_dict[deblurred_basename]
                    found = True
                    break
            if not found:
                print(f"Warning: No matching deblurred image for {gt_basename}, skipping...")
                continue
        else:
            deblurred_file = deblurred_dict[gt_basename]
        
        try:
            # Load images
            gt_img = load_img(gt_file)
            deblurred_img = load_img(deblurred_file)
            
            # Check if images have the same shape
            if gt_img.shape != deblurred_img.shape:
                print(f"Warning: Shape mismatch for {gt_basename}: "
                      f"GT {gt_img.shape} vs Deblurred {deblurred_img.shape}, resizing deblurred...")
                # Resize deblurred image to match ground truth
                deblurred_img = cv2.resize(deblurred_img, (gt_img.shape[1], gt_img.shape[0]), 
                                          interpolation=cv2.INTER_LINEAR)
            
            # Calculate metrics
            psnr = calculate_psnr(gt_img, deblurred_img, crop_border=crop_border, 
                                 input_order='HWC', test_y_channel=test_y_channel)
            ssim = calculate_ssim(gt_img, deblurred_img, crop_border=crop_border,
                                 input_order='HWC', test_y_channel=test_y_channel)
            
            # Store results
            result = {
                'filename': gt_basename,
                'gt_path': gt_file,
                'deblurred_path': deblurred_file,
                'psnr': float(psnr),
                'ssim': float(ssim)
            }
            results.append(result)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
        except Exception as e:
            print(f"Error processing {gt_basename}: {e}")
            continue
    
    # Calculate statistics
    if len(psnr_values) > 0:
        stats = {
            'total_images': len(results),
            'psnr': {
                'mean': float(np.mean(psnr_values)),
                'std': float(np.std(psnr_values)),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values)),
                'median': float(np.median(psnr_values))
            },
            'ssim': {
                'mean': float(np.mean(ssim_values)),
                'std': float(np.std(ssim_values)),
                'min': float(np.min(ssim_values)),
                'max': float(np.max(ssim_values)),
                'median': float(np.median(ssim_values))
            }
        }
    else:
        stats = {'total_images': 0, 'psnr': {}, 'ssim': {}}
        print("No images were successfully processed!")
        return None
    
    # Create final results dictionary
    final_results = {
        'gt_directory': gt_dir,
        'deblurred_directory': deblurred_dir,
        'crop_border': crop_border,
        'test_y_channel': test_y_channel,
        'statistics': stats,
        'per_image_results': results
    }
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Metrics Calculation Complete")
    print(f"{'='*60}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"\nPSNR Statistics:")
    print(f"  Mean:   {stats['psnr']['mean']:.4f} dB")
    print(f"  Std:    {stats['psnr']['std']:.4f} dB")
    print(f"  Min:    {stats['psnr']['min']:.4f} dB")
    print(f"  Max:    {stats['psnr']['max']:.4f} dB")
    print(f"  Median: {stats['psnr']['median']:.4f} dB")
    print(f"\nSSIM Statistics:")
    print(f"  Mean:   {stats['ssim']['mean']:.4f}")
    print(f"  Std:    {stats['ssim']['std']:.4f}")
    print(f"  Min:    {stats['ssim']['min']:.4f}")
    print(f"  Max:    {stats['ssim']['max']:.4f}")
    print(f"  Median: {stats['ssim']['median']:.4f}")
    print(f"\nResults saved to: {output_json}")
    print(f"{'='*60}")
    
    return final_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate PSNR and SSIM metrics')
    parser.add_argument('--gt_dir', type=str, 
                       default='Dataset/SloMoBlur/0_9999/groundtruth',
                       help='Directory containing ground truth images')
    parser.add_argument('--deblurred_dir', type=str,
                       default='Dataset/SloMoBlur/0_9999/deblurred',
                       help='Directory containing deblurred images')
    parser.add_argument('--output_json', type=str,
                       default='metrics_results.json',
                       help='Path to save results JSON file')
    parser.add_argument('--crop_border', type=int, default=0,
                       help='Cropped pixels in each edge (default: 0)')
    parser.add_argument('--test_y_channel', action='store_true',
                       help='Test on Y channel of YCbCr instead of RGB')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.gt_dir):
        raise ValueError(f"Ground truth directory not found: {args.gt_dir}")
    if not os.path.exists(args.deblurred_dir):
        raise ValueError(f"Deblurred directory not found: {args.deblurred_dir}")
    
    # Calculate metrics
    calculate_metrics(
        gt_dir=args.gt_dir,
        deblurred_dir=args.deblurred_dir,
        output_json=args.output_json,
        crop_border=args.crop_border,
        test_y_channel=args.test_y_channel
    )

