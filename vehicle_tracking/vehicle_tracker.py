from collections import OrderedDict
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from vehicle_tracking.vehicle_track import VehicleTrack
from vehicle_tracking.utils import iou_mat
from code_timing_profiling.profiling import profile
from code_timing_profiling.timing import timethis


class VehicleTracker:

    def __init__(self,
                 detection_vehicle_thresh,
                 inactive_steps_before_removed,
                 reid_iou_threshold,
                 max_traject_steps,
                 parking_ground,
                 cam="cam_1",
                 shape=(1280, 720)):

        #self.vehicle_detector = vehicle_detector
        self.detection_vehicle_thresh = detection_vehicle_thresh
        #self.regression_vehicle_thresh = regression_vehicle_thresh
        #self.detection_nms_thresh = detection_nms_thresh
        #self.regression_nms_thresh = regression_nms_thresh
        self.inactive_steps_before_removed = inactive_steps_before_removed
        self.reid_iou_threshold = reid_iou_threshold
        #self.do_align = do_align
        self.max_traject_steps = max_traject_steps
        self.parking_ground = parking_ground
        self.cam = cam
        self.shape = shape
        self.positions_mask = -1 * np.ones(shape=self.shape, dtype=np.int16)
        self.square_of_mask = OrderedDict() # Number of pixel (square) of each track id mask

        self.active_tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        #self.im_index = 0
        #self.results = {}

    def reset(self, hard=True):
        self.active_tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            #self.results = {}
            #self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.active_tracks = [t for t in self.active_tracks if t not in tracks]
        for t in tracks:
            t.bbox = t.traject_pos[-1]
        self.inactive_tracks += tracks

    def add_new_tracks(self, vehicle_detections_list):
        """Initiailizes new Track objectss and save them"""
        num_new = len(vehicle_detections_list)
        for i in range(num_new):
            self.active_tracks.append(VehicleTrack(score=vehicle_detections_list[i].score,
                                                   bbox=vehicle_detections_list[i].bbox,
                                                   positions=vehicle_detections_list[i].positions,
                                                   class_id=vehicle_detections_list[i].class_id,
                                                   track_id=self.track_num + i,
                                                   inactive_steps_before_removed=self.inactive_steps_before_removed,
                                                   max_traject_steps=self.max_traject_steps,
                                                   parking_ground=self.parking_ground,
                                                   cam=self.cam))
        self.track_num += num_new

    def get_active_boxes(self):
        if len(self.active_tracks) >= 1:
            bboxes = [t.bbox for t in self.active_tracks]
        else:
            bboxes = []
        return bboxes

    def get_active_scores(self):
        if len(self.active_tracks) >= 1:
            scores = [t.score for t in self.active_tracks]
        else:
            scores = []
        return scores

    def get_active_positions(self):
        if len(self.active_tracks) >= 1:
            positions = [t.positions for t in self.active_tracks]
        else:
            positions = []
        return positions

    def get_inactive_boxes(self):
        if len(self.inactive_tracks) >= 1:
            bboxes = [t.bbox for t in self.inactive_tracks]
        else:
            bboxes = []
        return bboxes

    def match_reid_iou_active(self, vehicle_detections):
        active_tracks_bboxes = self.get_active_boxes()
        detections_bboxes = [detection.bbox for detection in vehicle_detections]

        iou_matrix = iou_mat(active_tracks_bboxes, detections_bboxes)
        iou_matrix = np.where(iou_matrix > self.reid_iou_threshold, iou_matrix, 0)

        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

        matched_detection_indx = []
        matched_track_indx = []

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.reid_iou_threshold and self.active_tracks[r].class_id == vehicle_detections[c].class_id:
                matched_track_indx.append(r)
                matched_detection_indx.append(c)
                t = self.active_tracks[r]
                t.bbox = vehicle_detections[c].bbox
                t.score = vehicle_detections[c].score
                t.positions = vehicle_detections[c].positions
                t.time_stamp.append(time.time())
        unmatched_tracks = [self.active_tracks[i] for i in range(len(self.active_tracks)) if i not in matched_track_indx]
        unmatched_detections = [vehicle_detections[j] for j in range(len(vehicle_detections)) if j not in matched_track_indx]

        self.active_tracks = [self.active_tracks[k] for k in matched_track_indx]

        for track in unmatched_tracks:
            track.alive_time.append(time.time() - track.birth_time[-1])

        return unmatched_tracks, unmatched_detections

    def match_reid_iou_inactive(self, vehicle_detections):
        inactivate_tracks_bboxes = self.get_inactive_boxes()
        detections_bboxes = [detection.bbox for detection in vehicle_detections]

        iou_matrix = iou_mat(inactivate_tracks_bboxes, detections_bboxes)
        iou_matrix = np.where(iou_matrix > self.reid_iou_threshold, iou_matrix, 0)

        row_ind, col_ind = linear_sum_assignment(iou_matrix, maximize=True)

        matched_detection_idx = []
        matched_track_idx = []

        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.reid_iou_threshold and self.inactive_tracks[r].class_id == vehicle_detections[c].class_id: # Huhu đáng ra phải là self.inactive_tracks[r].class_id nhưng viết nhầm thành self.active_tracks[r].class_id
                matched_track_idx.append(r)
                matched_detection_idx.append(c)
                t = self.inactive_tracks[r]
                t.bbox = vehicle_detections[c].bbox
                t.score = vehicle_detections[c].score
                t.positions = vehicle_detections[c].positions
                t.reset_trajectory()
                t.inactive_steps = 0
                t.birth_time.append(time.time())
                self.active_tracks.append(t)

        matched_tracks = [self.inactive_tracks[i] for i in range(len(self.inactive_tracks)) if i in matched_track_idx]
        unmatched_detections = [vehicle_detections[j] for j in range(len(vehicle_detections)) if j not in matched_detection_idx]

        for t in matched_tracks:
            self.inactive_tracks.remove(t)

        return unmatched_detections

    def motion_step(self, track):
        track.bbox = track.bbox + track.traject_vel[-1] * (time.time() - track.time_stamp[-1])

    def motion(self):
        for t in self.active_tracks:
            if len(t.traject_pos) == 1:
                continue
            last_bboxes = t.traject_pos
            moments = t.time_stamp

            vs = np.asarray([(p2 - p1) / (t2 - t1) for p1, p2, t1, t2 in zip(list(last_bboxes), list(last_bboxes)[1:], list(moments), list(moments)[1:])], dtype=np.float)
            vs = np.mean(vs)

            t.traject_vel.append(vs)

            self.motion_step(t)

    @timethis
    def step(self, vehicle_detections):
        for t in self.active_tracks:
            if len(t.traject_pos) > 1:
                t.traject_pos.append(t.bbox.copy())
            else: # If len of t.traject_pos is 1 means that track just been created and t.bbox was already appended to t.traject_ps
                pass

        self.motion()

        if len(self.active_tracks) > 0 and len(vehicle_detections) > 0:
            unmatched_active_tracks, unmatched_detections = self.match_reid_iou_active(vehicle_detections=vehicle_detections)

            if len(unmatched_detections) > 0:
                if len(self.inactive_tracks) > 0: # Phải sửa từ active thành inactive
                    #print("1")
                    unmatched_detections = self.match_reid_iou_inactive(vehicle_detections=unmatched_detections)
                else:
                    #print("2")
                    pass
            else:
                #print("3")
                pass

            if len(unmatched_active_tracks) > 0:
                #print("4")
                self.tracks_to_inactive(unmatched_active_tracks)
            else:
                #print("5")
                pass

            if len(unmatched_detections) > 0:
                #print("6")
                self.add_new_tracks(vehicle_detections_list=unmatched_detections)
            else:
                #print("7")
                pass

        elif len(self.active_tracks) > 0 and len(vehicle_detections) == 0:
            #print("8")
            self.tracks_to_inactive(self.active_tracks)

        elif len(self.active_tracks) == 0 and len(vehicle_detections) > 0:
            if len(self.inactive_tracks) > 0:
                #print("9")
                unmatched_detections = self.match_reid_iou_inactive(vehicle_detections=vehicle_detections)
                if len(unmatched_detections) > 0:
                    #print("10")
                    self.add_new_tracks(vehicle_detections_list=unmatched_detections)
                else:
                    #print("11")
                    pass
            else:
                #print("12")
                self.add_new_tracks(vehicle_detections_list=vehicle_detections)
        else:
            #print("13")
            pass

        remove_inactive = []
        for t in self.inactive_tracks:
            t.inactive_steps += 1
            if t.inactive_steps > t.inactive_steps_before_removed:
                remove_inactive.append(t)

        for track in remove_inactive:
            self.inactive_tracks.remove(track)

        self.compute_positions_mask_and_square_mask_of_active_tracks()

    def get_result(self):
        return self.active_tracks

    def compute_positions_mask_and_square_mask_of_active_tracks(self):
        self.positions_mask = -1 * np.ones(shape=self.shape, dtype=np.int16)
        self.square_of_mask = OrderedDict()

        for t in self.active_tracks:
            rr, cc = t.positions
            self.positions_mask[rr, cc] = t.track_id
            self.square_of_mask[t.track_id] = rr.shape[0]

    def get_dict_convert_col_to_track_id(self):
        return dict(zip(list(range(len(self.square_of_mask.keys()))), list(self.square_of_mask.keys())))
