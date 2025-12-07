import os
import sys
import cv2
import argparse

from featurealgos.unified_features import (
    extract_rootsift_features,
    extract_lp_sift_features,
    extract_sift_daisy_features,
    extract_sift_phog_features,
    extract_phase_congruency_sift_features
)


def visualize_opencv_sift(image, output_path):
    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(image, None)

    if kps is None or len(kps) == 0:
        print(f"OpenCV SIFT: No keypoints detected")
        return 0

    img_with_kps = cv2.drawKeypoints(
        image, kps, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(output_path, img_with_kps)
    print(f"OpenCV SIFT: {len(kps)} keypoints, written to {output_path}")
    return len(kps)


def visualize_rootsift(image, output_path):
    kps, descs = extract_rootsift_features(image)

    if kps is None or len(kps) == 0:
        print(f"RootSIFT: No keypoints detected")
        return 0

    img_with_kps = cv2.drawKeypoints(
        image, kps, None,
        color=(255, 0, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(output_path, img_with_kps)
    print(f"RootSIFT: {len(kps)} keypoints, written to {output_path}")
    return len(kps)


def visualize_lp_sift(image, output_path):
    kps, descs = extract_lp_sift_features(image)

    if kps is None or len(kps) == 0:
        print(f"LP-SIFT: No keypoints detected")
        return 0

    img_with_kps = cv2.drawKeypoints(
        image, kps, None,
        color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(output_path, img_with_kps)
    print(f"LP-SIFT: {len(kps)} keypoints, written to {output_path}")
    return len(kps)


def visualize_sift_daisy(image, output_path):
    kps, descs = extract_sift_daisy_features(image)

    if kps is None or len(kps) == 0:
        print(f"SIFT+DAISY: No keypoints/descriptors generated")
        return 0

    img_with_kps = cv2.drawKeypoints(
        image, kps, None,
        color=(255, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(output_path, img_with_kps)
    print(f"SIFT+DAISY: {len(kps)} keypoints, written to {output_path}")
    return len(kps)


def visualize_sift_phog(image, output_path):
    kps, descs = extract_sift_phog_features(image)

    if kps is None or len(kps) == 0:
        print(f"SIFT+PHOG: No keypoints/descriptors generated")
        return 0

    img_with_kps = cv2.drawKeypoints(
        image, kps, None,
        color=(0, 255, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(output_path, img_with_kps)
    print(f"SIFT+PHOG: {len(kps)} keypoints, written to {output_path}")
    return len(kps)


def visualize_phase_congruency(image, output_path):
    kps, descs = extract_phase_congruency_sift_features(image)

    if kps is None or len(kps) == 0:
        print(f"PhaseCongruency: No keypoints/descriptors generated")
        return 0

    img_with_kps = cv2.drawKeypoints(
        image, kps, None,
        color=(255, 128, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(output_path, img_with_kps)
    print(f"PhaseCongruency: {len(kps)} keypoints, written to {output_path}")
    return len(kps)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize keypoints from different feature detection algorithms'
    )
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--output', '-o', default='keypoint_visualizations',
                        help='Output directory for visualizations (default: keypoint_visualizations)')

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image: {args.image}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.image))[0]

    print(f"Processing image: {args.image}")
    print(f"Output directory: {args.output}")

    visualize_opencv_sift(
        image.copy(),
        os.path.join(args.output, f'{base_name}_opencv_sift.png')
    )

    visualize_sift_phog(
        image.copy(),
        os.path.join(args.output, f'{base_name}_sift_phog.png')
    )

    visualize_phase_congruency(
        image.copy(),
        os.path.join(args.output, f'{base_name}_phase_congruency.png')
    )

    visualize_rootsift(
        image.copy(),
        os.path.join(args.output, f'{base_name}_rootsift.png')
    )

    visualize_lp_sift(
        image.copy(),
        os.path.join(args.output, f'{base_name}_lp_sift.png')
    )

    visualize_sift_daisy(
        image.copy(),
        os.path.join(args.output, f'{base_name}_sift_daisy.png')
    )

    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
