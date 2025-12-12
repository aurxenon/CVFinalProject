import numpy as np
from PIL import Image

# Load images
img1_path = 'demo/restored/Single_Image_Defocus_Deblurring/couple_input.png'
img2_path = 'demo/restored/Single_Image_Defocus_Deblurring/couple.png'

# Load and convert to numpy arrays
img1 = Image.open(img1_path)
img2 = Image.open(img2_path)

img1_np = np.array(img1)
img2_np = np.array(img2)

print("="*60)
print(f"Image 1: {img1_path}")
print("="*60)
print(f"Shape: {img1_np.shape} (height, width, channels)")
print(f"Data type: {img1_np.dtype}")
print(f"Min value: {img1_np.min()}")
print(f"Max value: {img1_np.max()}")
print(f"Mean value: {img1_np.mean():.2f}")
print(f"Std deviation: {img1_np.std():.2f}")
print(f"Value range: [{img1_np.min()}, {img1_np.max()}]")

print("\n" + "="*60)
print(f"Image 2: {img2_path}")
print("="*60)
print(f"Shape: {img2_np.shape} (height, width, channels)")
print(f"Data type: {img2_np.dtype}")
print(f"Min value: {img2_np.min()}")
print(f"Max value: {img2_np.max()}")
print(f"Mean value: {img2_np.mean():.2f}")
print(f"Std deviation: {img2_np.std():.2f}")
print(f"Value range: [{img2_np.min()}, {img2_np.max()}]")

print("\n" + "="*60)
print("Sample pixel values (first 5x5 region):")
print("="*60)
print("\nImage 1 (couple_input.png) - top-left 5x5 region:")
if img1_np.ndim == 3:
    print(img1_np[:5, :5, :])
else:
    print(img1_np[:5, :5])

print("\nImage 2 (couple.png) - top-left 5x5 region:")
if img2_np.ndim == 3:
    print(img2_np[:5, :5, :])
else:
    print(img2_np[:5, :5])
