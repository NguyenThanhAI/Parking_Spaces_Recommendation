import sys
import os
import numpy as np
import cv2
from parking_spaces_assignment.vehicle_detector import VehicleDetector
from parking_spaces_assignment.parking_space import ParkingSpacesInitializer

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)


class Matcher(object):
    def __init__(self,
                 checkpoint_name="mask_rcnn_cars_and_vehicles_0008.h5",
                 active_cams=["cam_1"],
                 shape=(720, 1280),
                 config_json_path=os.path.join(ROOT_DIR, "parking_spaces_data/parking_spaces_unified_id_segmen_in_cameras.json")):
        self.active_cams = active_cams
        self.parking_space_initializer = ParkingSpacesInitializer(active_cams=self.active_cams,
                                                                  shape=shape,
                                                                  config_json_path=config_json_path)
        #self.detectors_list = []
        #for cam in self.active_cams:
        #    self.detectors_list.append(VehicleDetector(checkpoint_name=checkpoint_name,
        #                                               cam=cam))
        self.detector = VehicleDetector(checkpoint_name=checkpoint_name)
        self.parking_spaces_list = self.parking_space_initializer.initialize_parking_spaces()

    def frame_match(self, frame, cam="cam_1"):

        detections_list = self.detector(frame=frame, cam=cam)

        parking_spaces_in_cam = list(filter(lambda x: cam in list(x.positions.keys()), self.parking_spaces_list))

        col_to_det_id = dict(zip(list(range(len(detections_list))), list(map(lambda x: x.detection_id, detections_list))))

        row_to_unified_id = dict(zip(list(range(len(parking_spaces_in_cam))), list(map(lambda x: x.unified_id, parking_spaces_in_cam))))

        print(col_to_det_id, row_to_unified_id)

        detection_masks = np.stack(list(map(lambda x: x.mask, detections_list)), axis=0)
        parking_spaces_in_cam_mask = np.stack(list(map(lambda x: x.positions_mask[cam], parking_spaces_in_cam)), axis=0)

        intersection = np.logical_and(parking_spaces_in_cam_mask[:, np.newaxis, :, :], detection_masks[np.newaxis, :, :, :])
        intersection = np.count_nonzero(intersection, axis=(2, 3)).astype(np.float32)
        ios = intersection / np.count_nonzero(parking_spaces_in_cam_mask, axis=(1, 2))[:, np.newaxis]
        iov = intersection / np.count_nonzero(detection_masks, axis=(1, 2))[np.newaxis, :]
        union = np.logical_or(parking_spaces_in_cam_mask[:, np.newaxis, :, :], detection_masks[np.newaxis, :, :, :])
        union = np.count_nonzero(union, axis=(2, 3))
        iou = intersection / union
        return detections_list, parking_spaces_in_cam, ios, iov, iou

matcher = Matcher()
image = cv2.imread(os.path.join(ROOT_DIR, "test_object_detection_models/images/201909_20190914_1_2019-09-14_01-00-00_80912.jpg"))
vehicles, parking_spaces, ios, iov, iou = matcher.frame_match(image)
