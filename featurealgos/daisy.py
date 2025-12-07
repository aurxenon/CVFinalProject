import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


def _bilinear_sample_channels(arr_ch, y, x):
    _, H, W = arr_ch.shape

    y = np.clip(y, 0.0, H - 1.0)
    x = np.clip(x, 0.0, W - 1.0)

    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = min(y0 + 1, H - 1)
    x1 = min(x0 + 1, W - 1)

    wy = y - y0
    wx = x - x0

    top = (1.0 - wx) * arr_ch[:, y0, x0] + wx * arr_ch[:, y0, x1]
    bottom = (1.0 - wx) * arr_ch[:, y1, x0] + wx * arr_ch[:, y1, x1]
    return (1.0 - wy) * top + wy * bottom


def _compute_image_gradients(gray):
    kx = np.array([[1, -1]], dtype=np.float32)
    ky = np.array([[1], [-1]], dtype=np.float32)

    Ix = cv2.filter2D(gray, cv2.CV_32F, kx, borderType=cv2.BORDER_REPLICATE)
    Iy = cv2.filter2D(gray, cv2.CV_32F, ky, borderType=cv2.BORDER_REPLICATE)
    return Ix, Iy


def _compute_orientation_maps(gray, H):
    Ix, Iy = _compute_image_gradients(gray)

    H_img, W_img = gray.shape
    orientation_maps = np.empty((H, H_img, W_img), dtype=np.float32)

    theta_o = np.linspace(0.0, 2.0 * np.pi, H, endpoint=False).astype(
        np.float32
    )

    for k, th in enumerate(theta_o):
        proj = np.cos(th) * Ix + np.sin(th) * Iy
        orientation_maps[k] = np.maximum(proj, 0.0)

    return orientation_maps


def _compute_convolved_orientation_maps(orientation_maps, R, Q):
    H_bins, H_img, W_img = orientation_maps.shape

    sigmas = R * (np.arange(Q, dtype=np.float32) + 1.0) / (2.0 * Q)

    conv_maps = np.empty((Q, H_bins, H_img, W_img), dtype=np.float32)

    prev_sigma = 0.0
    tmp = orientation_maps.astype(np.float32).copy()

    for i, s in enumerate(sigmas):
        delta = float(np.sqrt(max(s * s - prev_sigma * prev_sigma, 0.0)))
        if delta > 0.0:
            for k in range(H_bins):
                tmp[k] = gaussian_filter(tmp[k], sigma=delta, mode="reflect")
        conv_maps[i] = tmp.copy()
        prev_sigma = s

    return conv_maps


def daisy_at_point(
    image,
    y,
    x,
    R=15,
    Q=3,
    T=8,
    H=8,
    normalize_eps=1e-10,
    rotation_invariant=False,
):
    img = np.asarray(image)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = gray.astype(np.float32)
    if gray.max() > 1.0:
        gray /= 255.0

    orientation_maps = _compute_orientation_maps(gray, H)
    conv_maps = _compute_convolved_orientation_maps(
        orientation_maps, R, Q
    )

    radii = R * (np.arange(Q, dtype=np.float32) + 1.0) / float(Q)
    alphas = 2.0 * np.pi * np.arange(T, dtype=np.float32) / float(T)

    parts = []

    if rotation_invariant:
        center_hist_raw = conv_maps[0, :, y, x]
        dominant_bin = int(np.argmax(center_hist_raw))
    else:
        dominant_bin = 0

    center_hist = conv_maps[0, :, y, x].astype(np.float32)
    if rotation_invariant:
        center_hist = np.roll(center_hist, -dominant_bin)

    nrm = np.linalg.norm(center_hist)
    if nrm > normalize_eps:
        center_hist = center_hist / nrm
    parts.append(center_hist)

    for ring_idx, r in enumerate(radii):
        ring_maps = conv_maps[ring_idx]
        for alpha in alphas:
            yy = float(y) + r * np.sin(alpha)
            xx = float(x) + r * np.cos(alpha)
            hvec = _bilinear_sample_channels(ring_maps, yy, xx)
            if rotation_invariant:
                hvec = np.roll(hvec, -dominant_bin)
            nrm = np.linalg.norm(hvec)
            if nrm > normalize_eps:
                hvec = hvec / nrm
            parts.append(hvec)

    return np.concatenate(parts, axis=0).astype(np.float32)
