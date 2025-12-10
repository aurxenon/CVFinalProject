import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


class SuperPointONNX:
    def __init__(self, model_path, conf_thresh=0.015, nms_dist=4, input_size=(240, 320)):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.nms_dist = nms_dist
        self.input_size = input_size

        self.session = ort.InferenceSession(str(model_path))

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def preprocess_image(self, image):
        original_size = image.shape[:2]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        resized = cv2.resize(gray, (self.input_size[1], self.input_size[0]))

        scale_x = original_size[1] / self.input_size[1]
        scale_y = original_size[0] / self.input_size[0]

        normalized = resized.astype(np.float32) / 255.0
        input_tensor = normalized[np.newaxis, np.newaxis, :, :]

        return input_tensor, (scale_x, scale_y), original_size

    def run_inference(self, input_tensor):
        outputs = self.session.run(
            self.output_names, {self.input_name: input_tensor})
        return outputs[0], outputs[1], outputs[2]

    def extract_keypoints(self, scores, scale):
        if len(scores.shape) == 3:
            scores = scores[0]

        mask = scores > self.conf_thresh

        if self.nms_dist > 0:
            scores_nms = self.apply_nms(scores, self.nms_dist)
            mask = mask & (scores_nms > self.conf_thresh)

        pts = np.where(mask)
        keypoint_coords = np.stack([pts[1], pts[0]], axis=-1)
        confidences = scores[pts[0], pts[1]]

        keypoints = []
        for (x, y), conf in zip(keypoint_coords, confidences):
            x_scaled = float(x * scale[0])
            y_scaled = float(y * scale[1])
            kp = cv2.KeyPoint(x_scaled, y_scaled, 8, -1, float(conf))
            keypoints.append(kp)

        return keypoints, keypoint_coords

    def apply_nms(self, scores, nms_dist):
        kernel = np.ones((nms_dist * 2 + 1, nms_dist * 2 + 1),
                         dtype=np.float32)
        max_scores = cv2.dilate(scores, kernel)

        nms_mask = (scores == max_scores)
        nms_scores = scores * nms_mask.astype(np.float32)

        return nms_scores

    def extract_descriptors(self, descriptors, keypoint_coords):
        if len(descriptors.shape) == 4:
            descriptors = descriptors[0]

        desc_h, desc_w = descriptors.shape[1], descriptors.shape[2]
        cell_size = 8
        desc_coords = keypoint_coords // cell_size

        desc_coords[:, 0] = np.clip(desc_coords[:, 0], 0, desc_w - 1)
        desc_coords[:, 1] = np.clip(desc_coords[:, 1], 0, desc_h - 1)

        desc_coords = desc_coords.astype(np.int32)
        kp_descriptors = descriptors[:, desc_coords[:, 1], desc_coords[:, 0]]
        kp_descriptors = kp_descriptors.T

        return kp_descriptors

    def detect(self, image):
        input_tensor, scale, original_size = self.preprocess_image(image)

        scores, keypoint_logits, desc_map = self.run_inference(input_tensor)

        keypoints, keypoint_coords = self.extract_keypoints(scores, scale)

        descriptors = self.extract_descriptors(desc_map, keypoint_coords)

        return keypoints, descriptors
