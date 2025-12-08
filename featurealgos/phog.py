import numpy as np
import cv2


def _get_gray_image(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def build_integral_histogram(image, n_bins=20, angle=180):
    gray = _get_gray_image(image)
    h, w = gray.shape[:2]

    edges = cv2.Canny(gray, 100, 200)
    y_coords, x_coords = np.nonzero(edges)

    if len(y_coords) == 0:
        return np.zeros((h + 1, w + 1, n_bins), dtype=np.float32)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gx = grad_x[y_coords, x_coords]
    gy = grad_y[y_coords, x_coords]
    magnitude = np.sqrt(gx**2 + gy**2)

    if angle == 180:
        orientation = (np.arctan2(gy, gx) + np.pi/2) * 180 / np.pi
        orientation = np.mod(orientation, 180)
    else:
        orientation = (np.arctan2(gy, gx) + np.pi) * 180 / np.pi
        orientation = np.mod(orientation, 360)

    bin_width = angle / n_bins
    index_float = orientation / bin_width

    bin1 = np.floor(index_float).astype(int) % n_bins
    bin2 = (bin1 + 1) % n_bins

    f = index_float - np.floor(index_float)
    w1 = (1.0 - f) * magnitude
    w2 = f * magnitude

    H = np.zeros((h, w, n_bins), dtype=np.float32)

    H[y_coords, x_coords, bin1] += w1
    H[y_coords, x_coords, bin2] += w2

    integral_H = np.zeros((h + 1, w + 1, n_bins), dtype=np.float32)
    integral_H[1:, 1:] = H.cumsum(axis=0).cumsum(axis=1)

    return integral_H


def extract_batch_phog(integral_H, rects, L=3, n_bins=20):
    N = len(rects)
    if N == 0:
        d_len = n_bins * sum(4**l for l in range(L+1))
        return np.zeros((0, d_len), dtype=np.float32)

    W_patch = rects[0, 2]
    H_patch = rects[0, 3]

    x_origins = rects[:, 0]
    y_origins = rects[:, 1]

    max_h, max_w, _ = integral_H.shape

    level_descriptors = []

    for level in range(L + 1):
        n_cells = 2 ** level

        cell_h = H_patch // n_cells
        cell_w = W_patch // n_cells

        if cell_h < 1 or cell_w < 1:
            zeros = np.zeros((N, n_cells * n_cells * n_bins), dtype=np.float32)
            level_descriptors.append(zeros)
            continue

        grid_y, grid_x = np.meshgrid(
            np.arange(n_cells), np.arange(n_cells), indexing='ij')

        offset_y = (grid_y.flatten() * cell_h).astype(int)
        offset_x = (grid_x.flatten() * cell_w).astype(int)

        y_starts = y_origins[:, None] + offset_y[None, :]
        x_starts = x_origins[:, None] + offset_x[None, :]
        y_ends = y_starts + cell_h
        x_ends = x_starts + cell_w

        np.clip(y_starts, 0, max_h-1, out=y_starts)
        np.clip(x_starts, 0, max_w-1, out=x_starts)
        np.clip(y_ends, 0, max_h-1, out=y_ends)
        np.clip(x_ends, 0, max_w-1, out=x_ends)

        term_D = integral_H[y_ends, x_ends]
        term_B = integral_H[y_starts, x_ends]
        term_C = integral_H[y_ends, x_starts]
        term_A = integral_H[y_starts, x_starts]

        histograms = term_D - term_B - term_C + term_A

        np.maximum(histograms, 0, out=histograms)

        level_descriptors.append(histograms.reshape(N, -1))

    final_features = np.hstack(level_descriptors)

    sums = np.sum(final_features, axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    final_features /= sums

    return final_features
