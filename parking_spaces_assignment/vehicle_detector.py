import sys
import os
import numpy as np
import cv2
from mrcnn import config
from mrcnn import utils
from mrcnn.model import MaskRCNN
from parking_spaces_assignment.vehicle_detection import VehicleDetection


class MaskRCNNConfig(config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.0

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

class VehicleDetector(object):
    def __init__(self, checkpoint_name="mask_rcnn_cars_and_vehicles_0008.h5", cam="cam_1"):

        PRETRAINED_DIR = os.path.join(ROOT_DIR, "test_object_detection_models")

        PRETRAINED_PATH = os.path.join(PRETRAINED_DIR, checkpoint_name)

        LOG_DIR = os.path.join(PRETRAINED_DIR, "logs")

        self.cam = cam

        self.model = MaskRCNN(mode="inference", config=MaskRCNNConfig(), model_dir=LOG_DIR)

        self.model.load_weights(filepath=PRETRAINED_PATH, by_name=True)

    def __call__(self, frame):
        rgb_frame = frame[:, :, ::-1]

        results = self.model.detect([rgb_frame], verbose=0)

        result = results[0]

        rois, scores, class_ids, masks = result["rois"], result["scores"], result["class_ids"], result["masks"]

        masks = np.transpose(masks, axes=(2, 0, 1))

        detections_list = []

        for i, (roi, score, class_id, mask) in enumerate(zip(rois, scores, class_ids, masks)):
            if score >= 0.0 and class_id in [1]:
                y_min, x_min, y_max, x_max = roi
                bbox = [x_min, y_min, x_max, y_max]
                detections_list.append(VehicleDetection(score, bbox, mask, class_id, id, self.cam))
        return detections_list


detector = VehicleDetector()
image = cv2.imread(os.path.join(ROOT_DIR, "test_object_detection_models/images/car-park.jpg"))
vehicles =  detector(image)