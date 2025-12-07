import cv2
import numpy as np

def compute_dominant_orientation(image, x, y, L):
    h, w = image.shape
    
    sigma = L / 6.0
    radius = int(round(3.0 * 1.5 * sigma))
    
    y_min = max(0, y - radius)
    y_max = min(h, y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(w, x + radius + 1)
    
    patch = image[y_min:y_max, x_min:x_max]
    
    if patch.size == 0:
        return 0.0

    dx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
    dy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=True)

    Y, X = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    center_y, center_x = (y - y_min), (x - x_min)
    weight = np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * ((1.5 * sigma)**2)))
    
    weighted_mag = mag * weight

    hist, _ = np.histogram(ang.flatten(), bins=36, range=(0, 360), weights=weighted_mag.flatten())
    
    dominant_bin = np.argmax(hist)
    
    return float(dominant_bin * 10.0 + 5.0)


def lp_sift_detect_and_compute(image, L_scales, alpha=1e-6, contrast_threshold=5.0):
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    nr, nc = gray_image.shape

    i_indices, j_indices = np.mgrid[0:nr, 0:nc]
    linear_term = (i_indices * nc + j_indices).astype(np.float32)
    
    M_n = gray_image.astype(np.float32) + linear_term * alpha

    keypoints = []

    for L in L_scales:
        step = L

        for i in range(0, nr - L + 1, step):
            for j in range(0, nc - L + 1, step):
                window = M_n[i:i+L, j:j+L]

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(window)
                
                if (max_val - min_val) < contrast_threshold:
                    continue

                y_max_local, x_max_local = max_loc
                global_x_max = int(j + x_max_local)
                global_y_max = int(i + y_max_local)
                
                angle_max = compute_dominant_orientation(M_n, global_x_max, global_y_max, L)
                
                keypoints.append(cv2.KeyPoint(
                    x=float(global_x_max),
                    y=float(global_y_max),
                    size=float(L),
                    angle=angle_max,
                    response=float(max_val)
                ))

                y_min_local, x_min_local = min_loc
                global_x_min = int(j + x_min_local)
                global_y_min = int(i + y_min_local)

                angle_min = compute_dominant_orientation(M_n, global_x_min, global_y_min, L)

                keypoints.append(cv2.KeyPoint(
                    x=float(global_x_min),
                    y=float(global_y_min),
                    size=float(L),
                    angle=angle_min,
                    response=float(min_val)
                ))

    unique_keypoints = []
    seen_features = set()

    for kp in keypoints:
        ident = (int(round(kp.pt[0])), int(round(kp.pt[1])), int(round(kp.size)))
        if ident not in seen_features:
            unique_keypoints.append(kp)
            seen_features.add(ident)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.compute(gray_image, unique_keypoints)

    return keypoints, descriptors