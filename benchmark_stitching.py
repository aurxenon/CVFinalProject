from featurealgos.unified_features import (
    extract_opencv_sift_features,
    extract_rootsift_features,
    extract_lp_sift_features,
    extract_sift_daisy_features,
    extract_sift_phog_features,
    extract_phase_congruency_sift_features,
    extract_opencv_orb_features,
    extract_superpoint_features,
)
import cv2
import numpy as np
import time
import os
import argparse

SUPERPOINT_MODEL_PATH = 'exported/superpoint_coco_dynamic.onnx'


def extract_superpoint_wrapper(image):
    if not os.path.exists(SUPERPOINT_MODEL_PATH):
        print(
            f"Warning: SuperPoint model not found at {SUPERPOINT_MODEL_PATH}")
        return [], None
    return extract_superpoint_features(image, SUPERPOINT_MODEL_PATH)


FEATURE_ALGORITHMS = {
    'SIFT': extract_opencv_sift_features,
    'RootSIFT': extract_rootsift_features,
    'ORB': extract_opencv_orb_features,
    'DAISY': extract_sift_daisy_features,
    'PHOG': extract_sift_phog_features,
    'LP-SIFT': extract_lp_sift_features,
    'PC+SIFT': extract_phase_congruency_sift_features,
    'SuperPoint': extract_superpoint_wrapper,
}


def stitch_and_evaluate(img1, img2, algorithm_name, extractor_func, ratio_thresh=0.75):
    start_time = time.time()

    img1_color = img1.copy()
    img2_color = img2.copy()
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kps1, descs1 = extractor_func(img1_color)
    kps2, descs2 = extractor_func(img2_color)

    if descs1 is None or descs2 is None or len(kps1) < 4 or len(kps2) < 4:
        return {
            'Algorithm': algorithm_name,
            'Time (s)': time.time() - start_time,
            'Total Features': 0,
            'Total Matches': 0,
            'Inliers': 0,
            'Inlier Ratio': 0.0,
            'Reprojection Error': np.inf,
        }

    if descs1.dtype == np.uint8 and descs2.dtype == np.uint8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(descs1, descs2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    total_matches = len(good_matches)

    if total_matches < 4:
        return {
            'Algorithm': algorithm_name,
            'Time (s)': time.time() - start_time,
            'Total Features': len(kps1) + len(kps2),
            'Total Matches': total_matches,
            'Inliers': 0,
            'Inlier Ratio': 0.0,
            'Reprojection Error': np.inf,
        }

    pts1 = np.float32(
        [kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32(
        [kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    inliers = int(np.sum(mask))
    inlier_ratio = inliers / total_matches if total_matches > 0 else 0.0

    if inliers > 0 and H is not None:
        inlier_pts1 = pts1[mask.ravel() == 1]
        inlier_pts2 = pts2[mask.ravel() == 1]

        reprojected_pts2 = cv2.perspectiveTransform(inlier_pts1, H)

        errors = np.linalg.norm(reprojected_pts2 - inlier_pts2, axis=2)
        mean_reproj_error = np.mean(errors)
    else:
        mean_reproj_error = np.inf

    end_time = time.time()

    return {
        'Algorithm': algorithm_name,
        'Time (s)': end_time - start_time,
        'Total Features': len(kps1) + len(kps2),
        'Total Matches': total_matches,
        'Inliers': inliers,
        'Inlier Ratio': inlier_ratio,
        'Reprojection Error': mean_reproj_error,
    }


def main(image_dir="./herzjesu_dense/urd"):
    img_files = sorted([f for f in os.listdir(image_dir)
                       if f.lower().endswith(('.jpg', '.png'))])

    if len(img_files) < 2:
        print(
            f"\nERROR: Found only {len(img_files)} images. Need at least 2 for matching.")
        return

    pair1_name = img_files[0]
    pair2_name = img_files[1]

    img1 = cv2.imread(os.path.join(image_dir, pair1_name))
    img2 = cv2.imread(os.path.join(image_dir, pair2_name))

    if img1 is None or img2 is None:
        print(
            f"\nERROR: Could not load image files {pair1_name} or {pair2_name}.")
        return

    print(
        f"Testing Pair: {pair1_name} vs {pair2_name} | Resolution: {img1.shape[0]}x{img1.shape[1]}")

    results = []

    for name, func in FEATURE_ALGORITHMS.items():
        print(f"Running {name}...")
        result = stitch_and_evaluate(img1, img2, name, func)
        results.append(result)

    header = f"{'Algorithm':<15} | {'Time (s)':<10} | {'Features':<10} | {'Matches':<10} | {'Inliers':<10} | {'Inlier Ratio':<12} | {'Reproj. Error':<13}"
    print(header)
    print("-" * 80)

    for r in results:
        features_str = f"{r['Total Features']/2:.0f}x2"
        inlier_ratio_str = f"{r['Inlier Ratio']:.3f}"
        reproj_error_str = f"{r['Reprojection Error']:.2f}" if r['Reprojection Error'] < 1000 else "---"

        row = (
            f"{r['Algorithm']:<15} | "
            f"{r['Time (s)']:<10.3f} | "
            f"{features_str:<10} | "
            f"{r['Total Matches']:<10} | "
            f"{r['Inliers']:<10} | "
            f"{inlier_ratio_str:<12} | "
            f"{reproj_error_str:<13}"
        )
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
