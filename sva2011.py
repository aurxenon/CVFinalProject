import cv2
import numpy as np
import time
import os
import glob
from collections import defaultdict
import sys
from math import inf

OUTPUT_DIR = "Panoramas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUPERPOINT_MODEL_PATH = "exported/superpoint_coco_dynamic.onnx"

from featurealgos.unified_features import (
    extract_opencv_sift_features, extract_rootsift_features, extract_opencv_orb_features,
    extract_sift_daisy_features, extract_sift_phog_features, extract_phase_congruency_sift_features,
    extract_lp_sift_features, extract_superpoint_features
)

def extract_superpoint_wrapper(image):
    if not os.path.exists(SUPERPOINT_MODEL_PATH):
        print(f"Warning: SuperPoint model not found at {SUPERPOINT_MODEL_PATH}")
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

SVA_SEQUENCES = {
    "S1-ABC": ["a.jpg", "b.jpg", "c.jpg"],
    "S2-P101037": [f"P101037{i}.JPG" for i in range(1, 4)],
    "S3-P101037": [f"P101037{i}.JPG" for i in range(8, 10)],
    "S4-P10110": [f"P10110{i}.JPG" for i in range(69, 72)],
    "S5-P101137": [f"P101137{i}.JPG" for i in range(0, 2)],
}

def find_matches_and_homography(img1, img2, extractor_func, ratio_thresh=0.75):    
    kps1, descs1 = extractor_func(img1)
    kps2, descs2 = extractor_func(img2)
    
    if descs1 is None or descs2 is None or len(kps1) < 4 or len(kps2) < 4:
        return None, 0, 0, inf, "Not enough features found."

    if descs1.dtype == np.uint8 and descs2.dtype == np.uint8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        if descs1.dtype == np.uint8:
            matches = matcher.knnMatch(descs1, descs2, k=2)
        else:
            matches = matcher.knnMatch(descs1.astype(np.float32), descs2.astype(np.float32), k=2)
    except cv2.error:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        try:
            matches = matcher.knnMatch(descs1.astype(np.float32), descs2.astype(np.float32), k=2)
        except Exception:
            return None, 0, 0, inf, "Matching Error"

    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    total_matches = len(good_matches)
    
    if total_matches < 4:
        return None, 0, total_matches, inf, "RANSAC Input Failed (< 4 matches)"

    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    inliers = int(np.sum(mask))
    inlier_ratio = inliers / total_matches
    
    if inliers < 4 or H is None:
        return H, inlier_ratio, total_matches, inf, "RANSAC Inlier Count Failed (< 4 inliers)"

    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]
    reprojected_pts2 = cv2.perspectiveTransform(inlier_pts1, H)
    mre = np.mean(np.linalg.norm(reprojected_pts2 - inlier_pts2, axis=2))

    return H, inlier_ratio, total_matches, mre, "SUCCESS"


def stitch_sequence_and_benchmark(image_paths, seq_name, algo_name, extractor_func):    
    img_list = [cv2.imread(p) for p in image_paths]
    final_panorama = img_list[0].copy()
    H_cumulative = np.identity(3, dtype=np.float64)
    
    metrics_list = []
    total_start_time = time.time()

    for i in range(len(img_list) - 1):
        img_i = img_list[i]
        img_iplus1 = img_list[i+1]
        
        step_start_time = time.time()
        H_i_to_iplus1, ratio, matches, mre, status = find_matches_and_homography(
            img_i, img_iplus1, extractor_func
        )
        step_time = time.time() - step_start_time
        
        metrics_list.append({'ratio': ratio, 'mre': mre, 'time': step_time, 'status': status})
        
        if status != "SUCCESS":
            print(f"  Step {i} failed: {status}. Stopping stitch.")
            H = final_panorama.shape[0]
            W = final_panorama.shape[1]
            fail_img = np.zeros((H, W, 3), dtype=np.uint8)
            cv2.putText(fail_img, f"FAIL: {status}", (20, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return metrics_list, fail_img

        H_iplus1_to_i = np.linalg.inv(H_i_to_iplus1)
        
        H_cumulative = H_cumulative @ H_iplus1_to_i

        h_ref, w_ref = final_panorama.shape[:2]
        h_next, w_next = img_iplus1.shape[:2]

        pts_corners = np.float32([[0, 0], [w_next, 0], [w_next, h_next], [0, h_next]]).reshape(-1, 1, 2)
        pts_warped = cv2.perspectiveTransform(pts_corners, H_cumulative)
        
        [xmin, ymin] = np.int32(pts_warped.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts_warped.max(axis=0).ravel() + 0.5)
        
        t = [-xmin, -ymin]
        H_translate = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        w_final = max(w_ref + t[0], xmax - xmin)
        h_final = max(h_ref + t[1], ymax - ymin)

        MAX_SIZE = 32000
        if w_final > MAX_SIZE or h_final > MAX_SIZE:
            print(f"  Canvas too large ({w_final}x{h_final}), skipping this step")
            return metrics_list, final_panorama

        H_warp_next = H_translate @ H_cumulative
        H_warp_ref = H_translate @ np.identity(3)

        new_canvas = cv2.warpPerspective(final_panorama, H_warp_ref, (w_final, h_final))
        warped_next = cv2.warpPerspective(img_iplus1, H_warp_next, (w_final, h_final))

        H_cumulative = H_warp_next.copy()
        final_panorama = new_canvas

        mask = (warped_next > 0)
        final_panorama[mask] = warped_next[mask]
        
        print(f"  Step {i} successful (Inliers: {int(ratio*matches)}).")

    return metrics_list, final_panorama

def run_sva_benchmark_and_stitch(base_dir="/Users/aurxenon/Downloads/SVA2011/img/"):    
    full_sequence_results = defaultdict(dict)
    
    for seq_name, file_list in SVA_SEQUENCES.items():
        print(f"\n--- Running Sequence: {seq_name} ({len(file_list)} images) ---")
        
        image_paths = [os.path.join(base_dir, f) for f in file_list]
        
        for algo_name, algo_func in FEATURE_ALGORITHMS.items():
            print(f"  [ALGO] Starting {algo_name}...")
            
            metrics_list, panorama_img = stitch_sequence_and_benchmark(
                image_paths, seq_name, algo_name, algo_func
            )
            
            pairs_tested = len(metrics_list)
            
            if pairs_tested > 0:
                avg_ratio = np.mean([m['ratio'] for m in metrics_list])
                avg_mre = np.mean([m['mre'] for m in metrics_list if m['mre'] != inf])
                avg_time = np.sum([m['time'] for m in metrics_list])
            else:
                avg_ratio, avg_mre, avg_time = 0.0, inf, 0.0
            
            
            output_path = os.path.join(OUTPUT_DIR, f"{seq_name}_{algo_name}_Panorama.jpg")
            cv2.imwrite(output_path, panorama_img)
            
            print(f"  [RESULT] Saved {algo_name} panorama to {output_path}")

            full_sequence_results[seq_name][algo_name] = {
                "Avg Inlier Ratio": avg_ratio,
                "Avg Reproj. Error (px)": avg_mre if avg_mre != inf else 999.99,
                "Avg Time (s)": avg_time,
                "Pairs Tested": pairs_tested,
                "Status": "SUCCESS" if np.all([m['status'] == 'SUCCESS' for m in metrics_list]) else "PARTIAL/FAIL"
            }

    return full_sequence_results


def generate_results_table(full_results):
    summary_data = []
    
    for seq_name, algo_data in full_results.items():
        for algo_name, data in algo_data.items():
            summary_data.append(data | {"Sequence": seq_name, "Algorithm": algo_name})

    header = "| Sequence | Algorithm | Avg Inlier Ratio | Avg Reproj. Error (px) | Total Time (s) | Pairs Tested | Status |"
    separator = "|:---|:---|:---:|:---:|:---:|:---:|:---:|"
    
    rows = [header, separator]
    
    for data in summary_data:
        ratio_str = f"{data['Avg Inlier Ratio']:.3f}" if data['Pairs Tested'] > 0 else "N/A"
        mre_str = f"{data['Avg Reproj. Error (px)']:.2f}" if data['Avg Reproj. Error (px)'] < 999 else "FAIL"
        time_str = f"{data['Avg Time (s)']:.2f}"
        
        rows.append(
            f"| {data['Sequence']} | {data['Algorithm']} | {ratio_str} | {mre_str} | {time_str} | {data['Pairs Tested']} | {data['Status']} |"
        )
        
    return "\n".join(rows)


if __name__ == "__main__":
    results = run_sva_benchmark_and_stitch()
    
    final_table = generate_results_table(results)
    print(final_table)