import numpy as np


class VehicleDetection(object):
    def __init__(self, score, bbox, positions, class_id, detection_id, parking_ground, cam):
        self.score = score
        self.bbox = bbox
        self.positions = positions # Tập hợp các điểm [y1, y2, ..., yn], [x1, x2, ..., xn] nằm trong vehicle mask
        self.class_id = class_id
        self.detection_id = detection_id
        self.parking_ground = parking_ground
        self.cam = cam