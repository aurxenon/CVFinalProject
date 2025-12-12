from featurealgos.infer_onnx import SuperPointONNX
from featurealgos.phase_congruency import compute_sift_dominant_orientation
from featurealgos.phog import build_integral_histogram, extract_batch_phog
from featurealgos.daisy import daisy_at_point
import sys
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import argparse

PATCH_TYPES = ['ref', 'e1', 'e2', 'e3', 'e4', 'e5',
               'h1', 'h2', 'h3', 'h4', 'h5',
               't1', 't2', 't3', 't4', 't5']


class HPatchesSequence:
    def __init__(self, base_path):
        self.name = os.path.basename(base_path)
        self.base = base_path

        ref_path = os.path.join(base_path, 'ref.png')
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference patches not found: {ref_path}")

        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        self.n_patches = ref_img.shape[0] // 65

        for patch_type in PATCH_TYPES:
            patch_path = os.path.join(base_path, f'{patch_type}.png')
            if os.path.exists(patch_path):
                img = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
                patches = np.split(img, self.n_patches)
                setattr(self, patch_type, patches)


def normalize_rootsift(desc):
    if desc is None:
        return None
    desc /= (np.linalg.norm(desc, ord=1) + 1e-7)
    desc = np.sqrt(desc)
    desc /= (np.linalg.norm(desc) + 1e-7)
    return desc


_superpoint_model = None


def load_superpoint_model(model_path):
    global _superpoint_model
    _superpoint_model = SuperPointONNX(
        model_path=model_path,
        conf_thresh=0.015,
        nms_dist=4,
        input_size=(64, 64)
    )
    return _superpoint_model


def extract_superpoint_descriptor_at_center(patch, model):
    input_tensor, scale, original_size = model.preprocess_image(patch)

    scores, keypoint_logits, desc_map = model.run_inference(input_tensor)

    if len(desc_map.shape) == 4:
        desc_map = desc_map[0]

    desc_h, desc_w = desc_map.shape[1], desc_map.shape[2]
    center_y = desc_h // 2
    center_x = desc_w // 2

    descriptor = desc_map[:, center_y, center_x]

    descriptor = descriptor / (np.linalg.norm(descriptor) + 1e-7)

    return descriptor.astype(np.float32)


def get_center_keypoint(patch_size=65, angle=0.0):
    center = patch_size / 2.0
    kp = cv2.KeyPoint()
    kp.pt = (center, center)
    kp.size = 2 * center / 5.303
    kp.angle = angle
    return kp


def extract_descriptor_at_center(patch, method_name, superpoint_model=None):
    if method_name == 'superpoint':
        return extract_superpoint_descriptor_at_center(patch, superpoint_model)

    if method_name in ['opencv-sift', 'lp-sift']:
        sift = cv2.SIFT_create()
        kp = get_center_keypoint()
        _, descs = sift.compute(patch, [kp])
        if descs is None or len(descs) == 0:
            return None
        return descs[0]

    elif method_name == 'orb':
        orb = cv2.ORB_create()
        kp = get_center_keypoint()
        _, descs = orb.compute(patch, [kp])
        if descs is None or len(descs) == 0:
            return None
        return descs[0].astype(np.float32)

    elif method_name == 'rootsift':
        sift = cv2.SIFT_create()
        kp = get_center_keypoint()
        _, descs = sift.compute(patch, [kp])
        if descs is None or len(descs) == 0:
            return None
        return normalize_rootsift(descs[0])

    elif method_name == 'daisy':
        daisy = cv2.xfeatures2d.DAISY_create(
            radius=15,
            q_radius=3,
            q_theta=8,
            q_hist=8,
            norm=cv2.xfeatures2d.DAISY_NRM_PARTIAL
        )

        kp = get_center_keypoint()

        _, descs = daisy.compute(patch, [kp])

        if descs is None or len(descs) == 0:
            return None
        return descs[0]
        '''
        center_idx = 32
        desc = daisy_at_point(patch, y=center_idx, x=center_idx, 
                              R=15, Q=3, T=8, H=8)
        
        return desc.flatten() if desc is not None else None
        '''

    elif method_name == 'phog':
        n_bins = 20
        integral_H = build_integral_histogram(patch, n_bins=n_bins, angle=180)

        rects = np.array([[0, 0, 65, 65]], dtype=int)

        desc = extract_batch_phog(integral_H, rects, L=3, n_bins=n_bins)
        return desc.flatten() if desc is not None and desc.size > 0 else None

    elif method_name == 'phase-congruency':
        center_idx = 32
        angle = compute_sift_dominant_orientation(
            patch, x=center_idx, y=center_idx, size=12)

        sift = cv2.SIFT_create()
        kp = get_center_keypoint(angle=angle)
        _, descs = sift.compute(patch, [kp])

        if descs is None or len(descs) == 0:
            return None
        return descs[0]

    else:
        raise ValueError(f"Unknown method: {method_name}")


def extract_descriptors_for_method(hpatches_path, output_path, method_name, default_size=128, superpoint_model_path=None):
    superpoint_model = None
    if method_name == 'superpoint':
        superpoint_model = load_superpoint_model(superpoint_model_path)

    seq_dirs = sorted(glob.glob(os.path.join(hpatches_path, '*')))
    seq_dirs = [d for d in seq_dirs if os.path.isdir(d) and
                (os.path.basename(d).startswith('i_') or os.path.basename(d).startswith('v_'))]

    method_output_dir = os.path.join(output_path, method_name)
    os.makedirs(method_output_dir, exist_ok=True)

    for seq_path in tqdm(seq_dirs, desc=f"Processing sequences ({method_name})"):
        seq = HPatchesSequence(seq_path)

        seq_output_dir = os.path.join(method_output_dir, seq.name)
        os.makedirs(seq_output_dir, exist_ok=True)

        for patch_type in PATCH_TYPES:
            output_file = os.path.join(seq_output_dir, f'{patch_type}.csv')

            if os.path.exists(output_file):
                continue

            if not hasattr(seq, patch_type):
                continue

            patches = getattr(seq, patch_type)
            descriptors = []

            current_desc_size = default_size

            for patch in patches:
                desc = extract_descriptor_at_center(
                    patch, method_name, superpoint_model=superpoint_model)

                if desc is not None and len(desc) > 0:
                    current_desc_size = len(desc)
                else:
                    desc = np.zeros(current_desc_size, dtype=np.float32)

                descriptors.append(desc)

            descriptors = np.array(descriptors, dtype=np.float32)
            np.savetxt(output_file, descriptors, delimiter=',', fmt='%10.5f')


def main():
    parser = argparse.ArgumentParser(
        description='Extract descriptors for HPatches benchmark'
    )
    parser.add_argument('hpatches_path',
                        help='Path to hpatches-release folder')
    parser.add_argument('--output', '-o', default='hpatches_descriptors',
                        help='Output directory for descriptors')
    parser.add_argument('--methods', nargs='+',
                        choices=['all', 'opencv-sift', 'orb', 'rootsift', 'lp-sift',
                                 'daisy', 'phog', 'phase-congruency', 'superpoint'],
                        default=['all'],
                        help='Methods to extract')
    parser.add_argument('--superpoint-model', type=str,
                        help='Path to SuperPoint ONNX model')

    args = parser.parse_args()

    if not os.path.exists(args.hpatches_path):
        print(f"Error: HPatches path not found: {args.hpatches_path}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    methods_map = {
        'opencv-sift':      128,
        'orb':              32,
        'superpoint':       256,
        'rootsift':         128,
        'lp-sift':          128,
        'daisy':            200,
        'phog':             1700,
        'phase-congruency': 128
    }

    if 'all' in args.methods:
        methods_to_run = methods_map.keys()
    else:
        methods_to_run = args.methods

    if 'superpoint' in methods_to_run and args.superpoint_model is None:
        print("Error: --superpoint-model is required when using 'superpoint' method")
        sys.exit(1)

    print(f"\nExtracting descriptors from: {args.hpatches_path}")
    print(f"Output directory: {args.output}")
    print(f"Methods: {', '.join(methods_to_run)}\n")

    for method in methods_to_run:
        default_size = methods_map.get(method, 128)
        extract_descriptors_for_method(
            args.hpatches_path,
            args.output,
            method,
            default_size,
            superpoint_model_path=args.superpoint_model
        )
        print(f"\nCompleted {method}")

    print(f"Extraction complete!")


if __name__ == '__main__':
    main()
