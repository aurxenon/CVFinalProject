import cv2
import numpy as np


def extract_rootsift(image, eps=1e-7):
    sift = cv2.SIFT_create()

    kps, descs = sift.detectAndCompute(image, None)
    if descs is None or len(kps) == 0:
        return [], None

    descs /= (descs.sum(axis=1, keepdims=True) + eps)
    descs = np.sqrt(descs)
    descs /= (np.linalg.norm(descs, axis=1, keepdims=True) + eps)

    return kps, descs
