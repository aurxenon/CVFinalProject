import cv2
import numpy as np
from .root_sift import extract_rootsift
from .lp_sift import lp_sift_detect_and_compute
from .daisy import daisy_at_point
from .phog import compute_local_phog
from .phase_congruency import phase_congruency, compute_sift_dominant_orientation


def extract_rootsift_features(image):
    kps, descs = extract_rootsift(image)
    return kps, descs


def extract_lp_sift_features(image, L_scales=None):
    if L_scales is None:
        L_scales = [32, 64, 128]

    kps, descs = lp_sift_detect_and_compute(image, L_scales, alpha=1e-6)
    return kps, descs


def extract_sift_daisy_features(image):
    sift = cv2.SIFT_create()
    kps = sift.detect(image, None)

    if kps is None or len(kps) == 0:
        return [], None

    valid_kps = []
    descriptors = []

    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        descriptor = daisy_at_point(image, y=y, x=x, R=15, Q=3, T=8, H=8)

        if descriptor is not None and descriptor.size > 0:
            valid_kps.append(kp)
            descriptors.append(descriptor.flatten())

    if len(valid_kps) == 0:
        return [], None

    descs = np.array(descriptors, dtype=np.float32)
    return valid_kps, descs


def extract_sift_phog_features(image, patch_size=64):
    sift = cv2.SIFT_create()
    kps = sift.detect(image, None)

    if kps is None or len(kps) == 0:
        return [], None

    valid_kps = []
    descriptors = []

    half_size = patch_size // 2
    padded_image = cv2.copyMakeBorder(
        image, half_size, half_size, half_size, half_size,
        cv2.BORDER_CONSTANT, value=0
    )

    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))

        padded_x = x + half_size
        padded_y = y + half_size

        y_top = padded_y - half_size
        y_bottom = padded_y + half_size
        x_left = padded_x - half_size
        x_right = padded_x + half_size

        patch = padded_image[y_top:y_bottom, x_left:x_right]

        descriptor = compute_local_phog(patch, n_bins=20, angle=180, L=3)

        if descriptor is not None and descriptor.size > 0:
            valid_kps.append(kp)
            descriptors.append(descriptor.flatten())

    if len(valid_kps) == 0:
        return [], None

    descs = np.array(descriptors, dtype=np.float32)
    return valid_kps, descs


def extract_phase_congruency_sift_features(image):
    pc_map = phase_congruency(image)

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(pc_map, kernel)

    local_max = (pc_map == dilated) & (pc_map > 0)

    threshold = np.percentile(pc_map[local_max], 90) if np.any(local_max) else 0
    feature_points = np.argwhere((local_max) & (pc_map > threshold))

    if len(feature_points) == 0:
        assert False
        return [], None

    kps = []
    for y, x in feature_points:
        size = 10.0 + pc_map[y, x] * 20.0

        angle = compute_sift_dominant_orientation(image, x, y, size=size)

        kp = cv2.KeyPoint(x=float(x), y=float(y), size=size, angle=float(angle))
        kps.append(kp)

    sift = cv2.SIFT_create()
    kps, descs = sift.compute(image, kps)

    return kps, descs
