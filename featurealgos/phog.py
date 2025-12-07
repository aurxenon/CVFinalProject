import numpy as np
import cv2

def calculate_orientation_votes(image, angle=360, n_bins=8):
    edges = cv2.Canny(image, 100, 200)
    edge_mask = edges > 0

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    if angle == 180:
        orientation = (np.arctan2(grad_y, grad_x) + np.pi/2) * 180 / np.pi
        orientation[orientation < 0] += 180
        orientation[orientation >= 180] -= 180
    else:
        orientation = (np.arctan2(grad_y, grad_x) + np.pi) * 180 / np.pi
        orientation[orientation >= 360] -= 360

    edge_points = np.where(edge_mask)
    y_coords, x_coords = edge_points

    all_votes = []
    bin_width = angle / n_bins

    for y, x in zip(y_coords, x_coords):
        m = magnitude[y, x]
        if m == 0:
            continue

        ori = orientation[y, x]

        index_float = ori / bin_width

        bin1_float = np.floor(index_float)
        f = index_float - bin1_float

        bin1 = int(bin1_float) % n_bins
        bin2 = (bin1 + 1) % n_bins

        weight1 = 1.0 - f
        weight2 = f

        all_votes.append((bin1, m * weight1, y, x)) 
        all_votes.append((bin2, m * weight2, y, x))

    return all_votes


def phog_descriptor(votes, image_shape, L, n_bins):
    height, width = image_shape
    phog_vector = []

    for level in range(L + 1):
        n_cells = 2 ** level
        level_histogram = np.zeros(
            n_cells * n_cells * n_bins, dtype=np.float32)

        if n_cells == 1:
            cell_height = height
            cell_width = width
        else:
            cell_height = height // n_cells
            cell_width = width // n_cells

        for i in range(n_cells):
            y_start = i * cell_height
            y_end = height if i == n_cells - 1 else y_start + cell_height

            for j in range(n_cells):
                x_start = j * cell_width
                x_end = width if j == n_cells - 1 else x_start + cell_width

                cell_idx = i * n_cells + j

                for bin_idx, mag_vote, y, x in votes:
                    if y_start <= y < y_end and x_start <= x < x_end:
                        hist_index = cell_idx * n_bins + bin_idx
                        level_histogram[hist_index] += mag_vote

        phog_vector.append(level_histogram)

    p = np.concatenate(phog_vector)

    total = np.sum(p)
    if total != 0:
        p = p / total
        
    return p

def _get_gray_image(image):
    if len(image.shape) == 3:
        R, G, B = image[:, :, 2], image[:, :, 1], image[:, :, 0]
        gray = (0.3 * R + 0.59 * G + 0.11 * B).astype(np.uint8)
    else:
        gray = image.copy()
    return gray

def compute_local_phog(image_patch, n_bins=20, angle=180, L=3):
    gray = _get_gray_image(image_patch)
    image_shape = gray.shape[:2]

    votes = calculate_orientation_votes(gray, angle, n_bins)
    
    descriptor_length = n_bins
    for level in range(1, L + 1):
        descriptor_length += n_bins * (4 ** level)

    if not votes:
        return np.zeros(descriptor_length, dtype=np.float32)

    p = phog_descriptor(votes, image_shape, L, n_bins)
    
    return p
