import sys
import os
from collections import OrderedDict
import numpy as np
import cv2
from mrcnn import config
from mrcnn.model import MaskRCNN
from vehicle_tracking.vehicle_detection import VehicleDetection
from code_timing_profiling.profiling import profile
from code_timing_profiling.timing import timethis


class MaskRCNNConfig(config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.0

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)


class VehicleDetector(object):
    def __init__(self, checkpoint_name="mask_rcnn_cars_and_vehicles_0008.h5", detection_vehicle_thresh=0.4):

        PRETRAINED_DIR = os.path.join(ROOT_DIR, "test_object_detection_models")

        PRETRAINED_PATH = os.path.join(PRETRAINED_DIR, checkpoint_name)

        LOG_DIR = os.path.join(PRETRAINED_DIR, "logs")


        self.model = MaskRCNN(mode="inference", config=MaskRCNNConfig(), model_dir=LOG_DIR)

        self.model.load_weights(filepath=PRETRAINED_PATH, by_name=True)

        self.detection_vehicle_thresh = detection_vehicle_thresh

        self.positions_mask = OrderedDict()
        self.square_of_mask = OrderedDict() # Number of pixel (square) of each parking space unified id in correspondent camera

    @timethis
    def __call__(self, frame, parking_ground="parking_ground_SA", cam="cam_1"):

        self.positions_mask[cam] = -1 * np.ones(shape=frame.shape[:2], dtype=np.int16)
        self.square_of_mask[cam] = OrderedDict()

        rgb_frame = frame[:, :, ::-1]

        results = self.model.detect([rgb_frame], verbose=0)

        result = results[0]

        rois, scores, class_ids, masks = result["rois"], result["scores"], result["class_ids"], result["masks"]

        masks = np.transpose(masks, axes=(2, 0, 1))

        detections_list = []

        for det_id, (roi, score, class_id, mask) in enumerate(zip(rois, scores, class_ids, masks)):
            if score >= self.detection_vehicle_thresh and class_id in [1]:
                rr, cc = np.where(mask)
                self.positions_mask[cam][rr, cc] = det_id
                self.square_of_mask[cam][det_id] = rr.shape[0]
                y_min, y_max = np.min(rr), np.max(rr)
                x_min, x_max = np.min(cc), np.max(cc)
                bbox = [x_min, y_min, x_max, y_max]
                positions = np.array([rr, cc]) # Tập hợp các điểm [y1, y2, ..., yn], [x1, x2, ..., xn] nằm trong vehicle mask
                detections_list.append(VehicleDetection(score, bbox, positions, class_id, det_id, parking_ground, cam))
        return detections_list

    def get_dict_convert_col_to_det_id(self, cam):
        return dict(zip(list(range(len(self.square_of_mask[cam].keys()))), list(self.square_of_mask[cam].keys())))


#detector = VehicleDetector()
#image = cv2.imread(os.path.join(ROOT_DIR, "test_object_detection_models/images/car-park.jpg"))
#vehicles =  detector(image)
##print(detector.positions_mask)
#color_dict = {-1: 0}
#for i in range(len(vehicles)):
#    color_dict[i] = np.random.randint(100, 256, dtype=np.uint8)
#mask = detector.positions_mask["cam_1"]
#mask = np.vectorize(color_dict.get)(mask)
#mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3)).astype(np.uint8)
#
#cv2.imshow("Anh", image)
#cv2.waitKey(0)
#cv2.imshow("Mask", mask)
#cv2.waitKey()
#cv2.waitKey(0)
#cv2.destroyAllWindows()