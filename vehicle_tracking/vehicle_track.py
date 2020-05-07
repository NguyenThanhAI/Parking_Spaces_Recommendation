import time
from collections import deque
import numpy as np


class TentativeTrack(object):

    def __init__(self, score, bbox, positions, class_id, tentative_id, tentative_steps_before_accepted, max_traject_steps, parking_ground, cam):
        self.score = score
        self.bbox = bbox
        self.positions = positions
        self.class_id = class_id
        self.tentative_id = tentative_id
        self.tentative_steps = 0
        self.tentative_steps_before_accepted = tentative_steps_before_accepted
        self.traject_pos = deque([bbox.copy()], maxlen=max_traject_steps)
        self.traject_vel = deque()
        self.time_stamp = deque([time.time()], maxlen=max_traject_steps)
        self.parking_ground = parking_ground
        self.cam = cam


class VehicleTrack(object):

    def __init__(self, score, bbox, positions, class_id, track_id, inactive_steps_before_removed, traject_pos, traject_vel, time_stamp, parking_ground, cam):
        self.score = score
        self.bbox = bbox  # x_min, y_min, x_max, y_max
        self.positions = positions # Tập hợp các điểm [y1, y2, ..., yn], [x1, x2, ..., xn] nằm trong vehicle mask
        self.class_id = class_id
        self.track_id = track_id
        self.inactive_steps = 0
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.traject_pos = traject_pos.copy()
        self.traject_vel = traject_vel.copy()
        self.time_stamp = time_stamp.copy()
        self.birth_time = [time.time()]
        self.alive_time = []
        self.parking_ground = parking_ground
        self.cam = cam

    def has_positive_area(self):
        return self.bbox[2] > self.bbox[0] and self.bbox[3] > self.bbox[1]

    def reset_trajectory(self):
        self.traject_pos.clear()
        self.traject_pos.append(self.bbox.copy())
        self.traject_vel.clear()
        self.time_stamp.clear()
        self.time_stamp.append(time.time())
