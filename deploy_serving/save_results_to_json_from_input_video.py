import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from collections import OrderedDict

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

import numpy as np
import cv2

from parking_spaces_assignment.matcher import Matcher
from parking_spaces_assignment.pair import PairsScheduler
from vehicle_tracking.vehicle_tracker import VehicleTracker
from vehicle_tracking.vehicle_detection import VehicleDetection
from load_videos.videostream import QueuedStream
from parking_spaces_assignment.pair import TentativePair, Pair
from parking_spaces_assignment.utils import find_unique_values_and_frequency, write_information
from load_videos.videos_utils import get_time_amount_from_frames_number, get_start_time_from_video_name
import time


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class JsonPairsScheduler(PairsScheduler):
    def __init__(self,
                 time,
                 use_time_stamp,
                 active_cams,
                 database_dir="../database",
                 database_file=None,
                 use_mysql=False,
                 host="localhost",
                 user="Thanh",
                 passwd="Aimesoft",
                 reset_table=True,
                 save_to_db_period=2,
                 tentative_steps_before_accepted=30,
                 inactive_steps_before_removed=1000):
        super(JsonPairsScheduler, self).__init__(time=time,
                                                 use_time_stamp=use_time_stamp,
                                                 active_cams=active_cams,
                                                 database_dir=database_dir,
                                                 database_file=database_file,
                                                 use_mysql=use_mysql,
                                                 host=host,
                                                 user=user,
                                                 passwd=passwd,
                                                 reset_table=reset_table,
                                                 save_to_db_period=save_to_db_period,
                                                 tentative_steps_before_accepted=tentative_steps_before_accepted,
                                                 inactive_steps_before_removed=inactive_steps_before_removed)

    def step(self, uid_veh_list, num_frames, time_stamp, cam, results_json, frame_stride=1, fps=24):
        if not self.use_time_stamp:
            self.update_time_from_frame_numbers(num_frames, frame_stride, fps) # Cập nhật self.time
        else:
            self.time = time_stamp
        tentative_dict, active_dict, inactive_dict, deleted_dict = self.get_list_uid_veh_dict(cam=cam)
        inactive_to_active = {} # Những pair đang ở inactive list và ở bước này xuất hiện trong match pair giữa unified id và vehicle id
        active_this_step = {} # Những pair đang ở trong active list và bước này cũng xuất hiện trong match pair giữa unified id và vehicle id
        #active_to_inactive = [] # Những pair đang ở trong active list và bước này không xuất hiện trong match pair giữa unified id và vehicle id
        brand_new = {}
        new_tentatives = {}
        for uid_veh_id in uid_veh_list: # Xét từng uid_veh_id từ kết quả matcher trả về
            #existed = False
            if uid_veh_id in tentative_dict:
                tentative_dict[uid_veh_id].tentative_steps += 1
                if tentative_dict[uid_veh_id].tentative_steps > tentative_dict[uid_veh_id].tentative_steps_before_accepted:
                    brand_new[uid_veh_id] = Pair(unified_id=int(uid_veh_id[0]), vehicle_id=int(uid_veh_id[1]), class_id=int(uid_veh_id[2]), type_space=uid_veh_id[3], parking_ground=uid_veh_id[4], cam=uid_veh_id[5], birth_time=tentative_dict[uid_veh_id].birth_time, inactive_steps_before_removed=self.inactive_steps_before_removed)
                continue

            if uid_veh_id in active_dict: # Nếu uid_veh_id nằm trong active_dict nghĩa là cặp này đang active và tiếp tục bước này vẫn active
                #existed = True
                active_this_step[uid_veh_id] = active_dict[uid_veh_id]
                active_this_step[uid_veh_id].time_stamp_traject.append(self.time)
                continue

            if uid_veh_id in inactive_dict: # Nếu uid_veh_id nằm trong inactive_dict nghĩa là cặp này bước trước đang inactive bước này sẽ được active
                inactive_to_active[uid_veh_id] = inactive_dict[uid_veh_id]
                inactive_to_active[uid_veh_id].inactive_steps = 0 # Những cặp ở trong inactive bước trước, bước này thành active, số bước inactive được đưa về 0
                inactive_to_active[uid_veh_id].time_stamp_traject.append(self.time)
                continue

            if uid_veh_id in deleted_dict:
                print("{} in deleted_dict".format(uid_veh_id))

            new_tentatives[uid_veh_id] = TentativePair(unified_id=uid_veh_id[0], vehicle_id=uid_veh_id[1], class_id=uid_veh_id[2], type_space=uid_veh_id[3], parking_ground=uid_veh_id[4], cam=uid_veh_id[5], birth_time=self.time, tentative_steps_before_accepted=self.tentative_steps_before_accepted)

        tentative_dict = dict(filter(lambda x: x[0] not in brand_new, tentative_dict.items()))
        assert len(list(set(tentative_dict.keys()).intersection(brand_new.keys()))) == 0
        active_to_inactive = dict(filter(lambda x: x[0] not in active_this_step, active_dict.items())) # Những cặp active ở bước trước nhưng bước này không nằm trong list các cặp matcher trả về sẽ trở thành inactive ở bước này
        inactive_dict = dict(filter(lambda x: x[0] not in inactive_to_active, inactive_dict.items())) # Những inactive ở bước này bằng list inactive  loại bỏ đi những cặp trở thành active ở bước này

        assert len(list(set(tentative_dict.keys()).intersection(new_tentatives.keys()))) == 0
        tentative_dict = {**tentative_dict, **new_tentatives}
        self.verify_dicts(active_this_step, brand_new, inactive_to_active)
        active_dict = {**active_this_step, **brand_new, **inactive_to_active} # Tổng hợp lại, những cặp active ở bước này là hợp của những cặp ở trong active list sẵn và vẫn active ở bước này, những cặp mới hoàn toàn, đang là inactive thành active và đang deleted được rebrand
        assert len(list(set(inactive_dict.keys()).intersection(active_to_inactive.keys()))) == 0
        inactive_dict = {**inactive_dict, **active_to_inactive} # Những cặp inactive ở bước này là hợp của những cặp inactive vẫn inactive ở bước này và những cặp active ở bước trước thành inactive ở bước này

        # Xét các pair trong inactive_dict
        # Xem số inactive_step của từng pair lớn hơn ngưỡng thì thêm end_time, filled_period và chuyển sang delete_list
        inactive_to_deleted = {}
        for uid_veh_id in inactive_dict: # Duyệt từng phần tử của inactive list
            assert inactive_dict[uid_veh_id].unified_id == uid_veh_id[0] and inactive_dict[uid_veh_id].vehicle_id == uid_veh_id[1] and inactive_dict[uid_veh_id].class_id == uid_veh_id[2] and inactive_dict[uid_veh_id].type_space == uid_veh_id[3] and inactive_dict[uid_veh_id].parking_ground == uid_veh_id[4] and inactive_dict[uid_veh_id].cam == uid_veh_id[5] # Gọi pair instance ứng với uid_veh_id
            inactive_dict[uid_veh_id].inactive_steps += 1 # Tăng inactive step liên tiếp thêm 1
            if inactive_dict[uid_veh_id].inactive_steps > inactive_dict[uid_veh_id].inactive_steps_before_removed: # Nếu số bước liên tiếp là inactive của cặp này lớn hơn ngưỡng cho phép
                if not self.use_time_stamp:
                    num_seconds_inactive = get_time_amount_from_frames_number(num_frames=inactive_dict[uid_veh_id].inactive_steps, # Tính thời gian kể từ khi trở thành cặp inactive
                                                                              frame_stride=frame_stride,
                                                                              fps=fps)
                    time = self.time + timedelta(seconds=-num_seconds_inactive) # Thời gian trở thành căp inactive
                    inactive_dict[uid_veh_id].delete(time=time) # Tạo period là thời gian khởi tạo pair và thời điểm sang inactive??????
                else:
                    inactive_dict[uid_veh_id].delete(time=inactive_to_active[uid_veh_id].time_stamp_traject[-1])
                inactive_to_deleted[uid_veh_id] = inactive_dict[uid_veh_id] # List các cặp inactive thành deleted thêm phần tử này

        inactive_dict = dict(filter(lambda x: x[0] not in inactive_to_deleted, inactive_dict.items())) # Cập nhật lại list inactive bằng cách list inactive loại bỏ đi các inactive thành deleted
        deleted_dict.update(inactive_to_deleted) # Cập list các cặp bị deleted bằng cách cộng thêm các cặp ở trong list các cặp từ inactive thành deleted

        self.tentative_pairs[cam] = tentative_dict
        self.active_pairs[cam] = active_dict # Cập nhật lại self.active_pairs
        self.inactive_pairs[cam] = inactive_dict # Cập nhật lại self.inactive_pairs
        self.deleted_pairs[cam] = deleted_dict # Cập nhật lại self.deleted_pairs

        results_json[cam][num_frames]["tentative_pairs"] = {}
        for key in tentative_dict:
            results_json[cam][num_frames]["tentative_pairs"][str(key)] = {}
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["unified_id"] = tentative_dict[key].unified_id
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["vehicle_id"] = tentative_dict[key].vehicle_id
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["class_id"] = tentative_dict[key].class_id
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["type_space"] = tentative_dict[key].type_space
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["parking_ground"] = tentative_dict[key].parking_ground
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["cam"] = tentative_dict[key].cam
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["birth_time"] = tentative_dict[key].birth_time
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["tentative_steps"] = tentative_dict[key].tentative_steps
            results_json[cam][num_frames]["tentative_pairs"][str(key)]["tentative_steps_before_accepted"] = tentative_dict[key].tentative_steps_before_accepted

        results_json[cam][num_frames]["active_pairs"] = {}
        for key in active_dict:
            results_json[cam][num_frames]["active_pairs"][str(key)] = {}
            results_json[cam][num_frames]["active_pairs"][str(key)]["unified_id"] = active_dict[key].unified_id
            results_json[cam][num_frames]["active_pairs"][str(key)]["vehicle_id"] = active_dict[key].vehicle_id
            results_json[cam][num_frames]["active_pairs"][str(key)]["class_id"] = active_dict[key].class_id
            results_json[cam][num_frames]["active_pairs"][str(key)]["type_space"] = active_dict[key].type_space
            results_json[cam][num_frames]["active_pairs"][str(key)]["parking_ground"] = active_dict[key].parking_ground
            results_json[cam][num_frames]["active_pairs"][str(key)]["cam"] = active_dict[key].cam
            results_json[cam][num_frames]["active_pairs"][str(key)]["birth_time"] = active_dict[key].birth_time
            results_json[cam][num_frames]["active_pairs"][str(key)]["time_stamp_traject"] = active_dict[key].time_stamp_traject
            results_json[cam][num_frames]["active_pairs"][str(key)]["inactive_steps_before_removed"] = active_dict[key].inactive_steps_before_removed
            results_json[cam][num_frames]["active_pairs"][str(key)]["end_time"] = active_dict[key].end_time
            results_json[cam][num_frames]["active_pairs"][str(key)]["inactive_steps"] = active_dict[key].inactive_steps

        results_json[cam][num_frames]["inactive_pairs"] = {}
        for key in inactive_dict:
            results_json[cam][num_frames]["inactive_pairs"][str(key)] = {}
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["unified_id"] = inactive_dict[key].unified_id
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["vehicle_id"] = inactive_dict[key].vehicle_id
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["class_id"] = inactive_dict[key].class_id
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["type_space"] = inactive_dict[key].type_space
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["parking_ground"] = inactive_dict[key].parking_ground
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["cam"] = inactive_dict[key].cam
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["birth_time"] = inactive_dict[key].birth_time
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["time_stamp_traject"] = inactive_dict[key].time_stamp_traject
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["inactive_steps_before_removed"] = inactive_dict[key].inactive_steps_before_removed
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["end_time"] = inactive_dict[key].end_time
            results_json[cam][num_frames]["inactive_pairs"][str(key)]["inactive_steps"] = inactive_dict[key].inactive_steps

        results_json[cam][num_frames]["deleted_pairs"] = {}
        for key in deleted_dict:
            results_json[cam][num_frames]["deleted_pairs"][str(key)] = {}
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["unified_id"] = deleted_dict[key].unified_id
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["vehicle_id"] = deleted_dict[key].vehicle_id
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["class_id"] = deleted_dict[key].class_id
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["type_space"] = deleted_dict[key].type_space
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["parking_ground"] = deleted_dict[key].parking_ground
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["cam"] = deleted_dict[key].cam
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["birth_time"] = deleted_dict[key].birth_time
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["time_stamp_traject"] = deleted_dict[key].time_stamp_traject
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["inactive_steps_before_removed"] = deleted_dict[key].inactive_steps_before_removed
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["end_time"] = deleted_dict[key].end_time
            results_json[cam][num_frames]["deleted_pairs"][str(key)]["inactive_steps"] = deleted_dict[key].inactive_steps


        running_time = int((self.time - self.start_time).total_seconds())
        #print(running_time, running_time % self.save_to_db_period)
        if running_time > 0 and (running_time % self.save_to_db_period) == 0:
            self.save_pairs_to_db(cam=cam)



class JsonMatcher(Matcher):
    def __init__(self,
                 checkpoint_name="mask_rcnn_cars_and_vehicles_0008.h5",
                 model_arch="mask_rcnn",
                 cuda=False,
                 parking_ground="parking_ground_SA",
                 active_cams=["cam_2"],
                 shape=(720, 1280),
                 config_json_path=os.path.join(ROOT_DIR, "parking_spaces_data/parking_spaces_unified_id_segmen_in_cameras.json"),
                 detection_vehicle_thresh=0.4,
                 run_multiprocessing=True):
        super(JsonMatcher, self).__init__(checkpoint_name=checkpoint_name,
                                          model_arch=model_arch,
                                          cuda=cuda,
                                          parking_ground=parking_ground,
                                          active_cams=active_cams,
                                          shape=shape,
                                          config_json_path=config_json_path,
                                          detection_vehicle_thresh=detection_vehicle_thresh,
                                          run_multiprocessing=run_multiprocessing)

    def frame_match(self, frame, vehicles_list, results_json, frame_id, cam="cam_2", ios_threshold=0.1, iov_threshold=0.4, is_tracking=False, tracker=None):

        assert cam in self.active_cams, "{} must be in {} of Matcher".format(cam, self.active_cams)

        # vehicles_list = self.detector(frame=frame, parking_ground=self.parking_ground, cam=cam) # Phát hiện vehicle detection dưới dạng list các instance vehicle_detection

        # if is_tracking: # Nếu có sử dụng tracking
        #    assert tracker, "vehicles tracker cannot be None"
        #    tracker.step(vehicle_detections=vehicles_list)
        #    vehicles_list = tracker.get_result() #Sử dụng track lấy ra các vehicle track là list các instance của vehicle_track
        start = time.time()
        parking_spaces_in_cam = list(filter(lambda x: cam in list(x.positions.keys()),
                                            self.parking_spaces_list))  # Lọc ra các parking spaces có trong cam hiện tại dưới dạng list các instance parking_space
        outlier_parking_spaces_in_cam = list(filter(lambda x: cam in list(x.positions.keys()),
                                                    self.outlier_parking_spaces_list))  # Lọc ra các outlier parking spaces có trong cam hiện tại dưới dạng list các instance parking space
        # if not is_tracking: # Tạo ra dictionary map từ vehicle_id (detection_id nếu không track, track_id nếu sử dụng track), sang instance của vehicle_detection (nếu không sử dụng track) hoặc vehicle_track (nếu sử dụng track)
        vehicle_id_to_vehicle = dict(map(lambda x: (x.detection_id, x), vehicles_list))
        # else:
        #    vehicle_id_to_vehicle = dict(map(lambda x: (x.track_id, x), vehicles_list))

        unified_id_to_ps = dict(map(lambda x: (x.unified_id, x),
                                    parking_spaces_in_cam))  # Tạo ra dictionary map từ unified_id sang instance của parking_space
        outlier_unified_id_to_ps = dict(map(lambda x: (x.unified_id, x),
                                            outlier_parking_spaces_in_cam))  # Tạo ra dictionary map từ unified_id sang instance của parking_space (outlier parking space)

        # if is_tracking: # Lấy positions_mask của parking_space từ parkingspaceinitializer._positions_map[cam] và positions_mask của vehicle từ vehicle_detector.positions_mask[cam] (nếu không sử dụng track) hoặc vehicle_tracker.positions_mask (nếu sử dụng track
        #    vehicle_masks = tracker.positions_mask
        #    vehicle_square_of_mask = tracker.square_of_mask
        # else:
        vehicle_masks = self.positions_mask[cam]
        vehicle_square_of_mask = self.square_of_mask[cam]
        parking_spaces_in_cam_mask = self.parking_space_initializer.positions_mask[cam]
        # Tạo hai dictionary unified_id_to_vehicle_id_ios và vehicle_id_to_unified_id_ios chứa thông tin ios (intersection over space) {unified_id1: {vehicle_id1: ..., vehicle_id2: ..., ...}, unified_id2: ...}, {vehicle_id1: {unified_id1:..., unified_id2:...,...}, vehicle_id2:...}
        unified_id_to_vehicle_id_ios = {}
        vehicle_id_to_unified_id_ios = {}

        if len(parking_spaces_in_cam) < len(
                vehicles_list):  # Giả sử chọn vòng lặp for theo các key của dictionary của unified_id sang instance của parking_space:
            for unified_id in unified_id_to_ps:
                x_min, y_min, x_max, y_max = unified_id_to_ps[unified_id].bbox[
                    cam]  # Vì bbox là dict nên phải key cam vào một unified ứng với mỗi cam sẽ là một bbox khác nhau
                cropped_ps_mask = parking_spaces_in_cam_mask[y_min:y_max + 1,
                                  x_min:x_max + 1]  # Crop parking space positions mask
                cropped_vh_mask = vehicle_masks[y_min:y_max + 1, x_min:x_max + 1]  # Crop vehicle positions mask
                cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)
                # print("unified_id", unified_id, cropped_mask.shape)
                inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=unified_id,
                                                              use_unified_id=True)
                for ps_veh in inter_dict:
                    uid, vid = ps_veh
                    inter = inter_dict[ps_veh]
                    ios = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                    if ios < ios_threshold:  # Tìm số lượng giao của unified_id với các vehicle_id trong vùng crop trên và lưu vào 2 dictionary unified_id_to_vehicle_id và vehicle_id_to_unified_id nếu ios thỏa mãn > threshold đặt trước, nếu không thì bỏ qua
                        continue
                    else:
                        if uid not in unified_id_to_vehicle_id_ios:
                            unified_id_to_vehicle_id_ios[uid] = {}
                        if vid not in unified_id_to_vehicle_id_ios[uid]:
                            unified_id_to_vehicle_id_ios[uid][vid] = inter / \
                                                                     self.parking_space_initializer.square_of_mask[cam][
                                                                         uid]
                        else:
                            assert unified_id_to_vehicle_id_ios[uid][vid] == (
                                        inter / self.parking_space_initializer.square_of_mask[cam][
                                    uid]), "ios of 2 times is not equal"
                        if vid not in vehicle_id_to_unified_id_ios:
                            vehicle_id_to_unified_id_ios[vid] = {}
                        if uid not in vehicle_id_to_unified_id_ios[vid]:
                            vehicle_id_to_unified_id_ios[vid][uid] = inter / \
                                                                     self.parking_space_initializer.square_of_mask[cam][
                                                                         uid]
                        else:
                            assert vehicle_id_to_unified_id_ios[vid][uid] == (
                                        inter / self.parking_space_initializer.square_of_mask[cam][
                                    uid]), "ios of 2 times is not equal"
        else:
            for vehicle_id in vehicle_id_to_vehicle:
                x_min, y_min, x_max, y_max = vehicle_id_to_vehicle[vehicle_id].bbox
                cropped_ps_mask = parking_spaces_in_cam_mask[y_min:y_max + 1,
                                  x_min:x_max + 1]  # Crop parking space positions mask
                cropped_vh_mask = vehicle_masks[y_min:y_max + 1, x_min:x_max + 1]  # Crop vehicle positions mask
                cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)
                # print("vehicle_id", vehicle_id, cropped_mask.shape)
                inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=vehicle_id,
                                                              use_unified_id=False)
                for ps_veh in inter_dict:
                    uid, vid = ps_veh
                    inter = inter_dict[ps_veh]
                    ios = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                    if ios < ios_threshold:
                        continue
                    else:
                        if uid not in unified_id_to_vehicle_id_ios:
                            unified_id_to_vehicle_id_ios[uid] = {}
                        if vid not in unified_id_to_vehicle_id_ios[uid]:
                            unified_id_to_vehicle_id_ios[uid][vid] = inter / \
                                                                     self.parking_space_initializer.square_of_mask[cam][
                                                                         uid]
                        else:
                            assert unified_id_to_vehicle_id_ios[uid][vid] == (
                                        inter / self.parking_space_initializer.square_of_mask[cam][
                                    uid]), "ios of 2 times is not equal"
                        if vid not in vehicle_id_to_unified_id_ios:
                            vehicle_id_to_unified_id_ios[vid] = {}
                        if uid not in vehicle_id_to_unified_id_ios[vid]:
                            vehicle_id_to_unified_id_ios[vid][uid] = inter / \
                                                                     self.parking_space_initializer.square_of_mask[cam][
                                                                         uid]
                        else:
                            assert vehicle_id_to_unified_id_ios[vid][uid] == (
                                        inter / self.parking_space_initializer.square_of_mask[cam][
                                    uid]), "ios of 2 times is not equal"
        # print("unified_id_to_vehicle_id_ios: {}, vehicle_id_to_unified_id_ios: {}".format(unified_id_to_vehicle_id_ios, vehicle_id_to_unified_id_ios))
        results_json[cam][frame_id]["unified_id_to_vehicle_id_ios"] = unified_id_to_vehicle_id_ios
        results_json[cam][frame_id]["vehicle_id_to_unified_id_ios"] = vehicle_id_to_unified_id_ios
        end = time.time()
        # print("This block consumes {} seconds".format(end - start))
        unified_id_status_dict = dict(zip(list(unified_id_to_ps.keys()), ["available"] * len(list(
            unified_id_to_ps.keys()))))  # Tạo một unified_id_status_dict = {unified_id: "unknown", ....} tất cả các unified_id có trạng thái ban đầu là unknown # Từ filled thành unknown
        start = time.time()
        uid_veh_id_match_list = []
        matched_vehicles_id_list = []
        considered_vehicle_id_list = []  # Đặt một considered_vehicle_id_list = [] chứa các vehicle_id đã được xét với các unified_id
        for unified_id in unified_id_status_dict:  # Duyệt từng unified_id trên unified_id_status_dict:
            if unified_id not in unified_id_to_vehicle_id_ios:  # Nếu unified_id không là tồn tại là key trong unified_id_to_vehicle_id thì chuyển trạng thái của unified_id trong unified_id_status_dict là "available"
                unified_id_status_dict[unified_id] = "available"
            else:  # unified_id là tồn tại là key trong unified_id_to_vehicle_id
                for vehicle_id in unified_id_to_vehicle_id_ios[
                    unified_id]:  # Xét từng vehicle_id có ios giao trong unified_id_to_vehicle_id[unified_id]
                    if vehicle_id not in considered_vehicle_id_list:  # Nếu vehicle_id này không nằm trong considered_vehicle_id_list = [] thì xét tiếp, ngược lại đã ở trong rồi thì bỏ qua chuyển sang vehicle_id tiếp theo
                        assert unified_id in vehicle_id_to_unified_id_ios[
                            vehicle_id], "Parking space id {} must be in vehicle to parking space ios {}".format(
                            unified_id, vehicle_id)
                        if len(vehicle_id_to_unified_id_ios[
                                   vehicle_id]) == 1:  # Nếu vehicle_id_to_unified_id[vehicle_id] của vehicle_id đang xét này chỉ có đúng một unified_id đang xét
                            unified_id_status_dict[unified_id] = "filled"
                            uid_veh_id_match_list.append(
                                [unified_id, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id,
                                 unified_id_to_ps[unified_id].type_space, self.parking_ground, cam])
                            matched_vehicles_id_list.append(vehicle_id)
                            # print("Parking space unified id {} and vehicle id {} is matched".format(unified_id, vehicle_id))
                        else:  # Nếu vehicle_id_to_unified_id[vehicle_id] của vehicle_id đang xét nhiều hơn một unified_id
                            pspace_dict = {}  # Dictionary of dictionary, each dictionary represents and parking space with information east level, south level and adjacencies against orient
                            for uid_match in vehicle_id_to_unified_id_ios[
                                vehicle_id]:  # Xét từng unified_id này, tạo một pspace_dict lưu trữ các thông tin: south_level, east_level, visited, adjacencies, ios, reversed_considered_orients (tất nhiên phải tương ứng với cam)
                                uid_match_dict = {}
                                uid_match_dict["east_level"] = 0
                                uid_match_dict["south_level"] = 0
                                uid_match_dict["visited"] = False
                                pspace = unified_id_to_ps[uid_match]
                                adjacencies = dict(filter(lambda x: x[1] is not None, pspace.adjacencies.items()))
                                adjacencies = dict(filter(lambda x: x[1] in vehicle_id_to_unified_id_ios[vehicle_id],
                                                          adjacencies.items()))
                                uid_match_dict["adjacencies"] = adjacencies
                                uid_match_dict["ios"] = vehicle_id_to_unified_id_ios[vehicle_id][uid_match]
                                uid_match_dict["reversed_considered_orients"] = pspace.reversed_considered_orients[
                                    cam] if cam in pspace.reversed_considered_orients else {}
                                pspace_dict[uid_match] = uid_match_dict

                            trace = []

                            def traverse_neighbors(
                                    uid_match):  # Sử dụng đệ quy tương hỗ để tính south_level, east_level của từng unified_id
                                # pspace[uid_match]["visited"] = True
                                if "eastern_adjacency" in pspace_dict[uid_match]["adjacencies"]:
                                    traverse_orient(uid_match, orient="east")
                                if "western_adjacency" in pspace_dict[uid_match]["adjacencies"]:
                                    traverse_orient(uid_match, orient="west")
                                if "southern_adjacency" in pspace_dict[uid_match]["adjacencies"]:
                                    traverse_orient(uid_match, orient="south")
                                if "northern_adjacency" in pspace_dict[uid_match]["adjacencies"]:
                                    traverse_orient(uid_match, orient="north")
                                if "south_east_adjacency" in pspace_dict[uid_match]["adjacencies"]:
                                    traverse_orient(uid_match, orient="south_east")
                                if "south_west_adjacency" in pspace_dict[uid_match]["adjacencies"]:
                                    traverse_orient(uid_match, orient="south_west")
                                if "north_west_adjacency" in pspace_dict[uid_match]["adjacencies"]:
                                    traverse_orient(uid_match, orient="north_west")
                                if "north_east_adjacency" in pspace_dict[uid_match]["adjacencies"]:
                                    traverse_orient(uid_match, orient="north_east")

                            def traverse_orient(uid_match, orient):
                                if orient == "east":
                                    adjacency = "eastern_adjacency"
                                elif orient == "west":
                                    adjacency = "western_adjacency"
                                elif orient == "south":
                                    adjacency = "southern_adjacency"
                                elif orient == "north":
                                    adjacency = "northern_adjacency"
                                elif orient == "south_east":
                                    adjacency = "south_east_adjacency"
                                elif orient == "south_west":
                                    adjacency = "south_west_adjacency"
                                elif orient == "north_west":
                                    adjacency = "north_west_adjacency"
                                else:
                                    adjacency = "north_east_adjacency"

                                if not pspace_dict[pspace_dict[uid_match]["adjacencies"][adjacency]]["visited"]:
                                    east_level = pspace_dict[uid_match]["east_level"]
                                    south_level = pspace_dict[uid_match]["south_level"]
                                    uid_match = pspace_dict[uid_match]["adjacencies"][adjacency]
                                    if orient == "east":
                                        pspace_dict[uid_match]["east_level"] = east_level + 1
                                        pspace_dict[uid_match]["south_level"] = south_level
                                    elif orient == "west":
                                        pspace_dict[uid_match]["east_level"] = east_level - 1
                                        pspace_dict[uid_match]["south_level"] = south_level
                                    elif orient == "south":
                                        pspace_dict[uid_match]["east_level"] = east_level
                                        pspace_dict[uid_match]["south_level"] = south_level + 1
                                    elif orient == "north":
                                        pspace_dict[uid_match]["east_level"] = east_level
                                        pspace_dict[uid_match]["south_level"] = south_level - 1
                                    elif orient == "south_east":
                                        pspace_dict[uid_match]["east_level"] = east_level + 1
                                        pspace_dict[uid_match]["south_level"] = south_level + 1
                                    elif orient == "south_west":
                                        pspace_dict[uid_match]["east_level"] = east_level - 1
                                        pspace_dict[uid_match]["south_level"] = south_level + 1
                                    elif orient == "north_west":
                                        pspace_dict[uid_match]["east_level"] = east_level - 1
                                        pspace_dict[uid_match]["south_level"] = south_level - 1
                                    else:
                                        pspace_dict[uid_match]["east_level"] = east_level + 1
                                        pspace_dict[uid_match]["south_level"] = south_level - 1
                                    pspace_dict[uid_match]["visited"] = True
                                    trace.append(uid_match)
                                    traverse_neighbors(uid_match)
                                else:
                                    return

                            random_uid = np.random.choice(list(vehicle_id_to_unified_id_ios[vehicle_id].keys()))

                            pspace_dict[random_uid]["visited"] = True  # Choose random_uid as starting point
                            trace.append(random_uid)  # Initialize track
                            traverse_neighbors(random_uid)  # Assign value to east_level and south_level

                            # print("Random unified_id {}, trace {}".format(random_uid, trace))

                            reversed_considered_orients = {}  # Tạo reversed_considered_orients = {"orients": [unified_id1, unified_id2, ...], ....}
                            for uid_match in pspace_dict:
                                orients = pspace_dict[uid_match]["reversed_considered_orients"]
                                for orient in orients:
                                    if orient not in reversed_considered_orients:
                                        reversed_considered_orients[orient] = []
                                    reversed_considered_orients[orient].append(uid_match)
                            # print("Reversed_considered_orients {}".format(reversed_considered_orients))
                            # Xét các hướng nếu trong reversed_considered_orients có
                            considered_east_west_uid_list = []
                            max_east = False
                            if "north_east" in reversed_considered_orients or "east" in reversed_considered_orients \
                                    or "south_east" in reversed_considered_orients:  # Tây Nam, Tây Bắc, Tây thì ưu tiên chọn điểm đỗ ở cực Đông
                                min_east_level = \
                                pspace_dict[min(pspace_dict.keys(), key=lambda x: pspace_dict[x]["east_level"])][
                                    "east_level"]
                                considered_uid = list(dict(
                                    filter(lambda x: x[1]["east_level"] == min_east_level, pspace_dict.items())).keys())
                                considered_east_west_uid_list.extend(considered_uid)
                                # print("min_east_level {}, consider_uid {}, considered_east_west_uid_list {}, max_east {}".format(min_east_level, considered_uid, considered_east_west_uid_list, max_east))

                            elif "north_west" in reversed_considered_orients or "west" in reversed_considered_orients \
                                    or "south_west" in reversed_considered_orients:  # Đông Nam, Đông Bắc, Đông thì ưu tiên chọn điểm đỗ ở cực Tây
                                max_east_level = \
                                pspace_dict[max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["east_level"])][
                                    "east_level"]
                                considered_uid = list(dict(
                                    filter(lambda x: x[1]["east_level"] == max_east_level, pspace_dict.items())).keys())
                                considered_east_west_uid_list.extend(considered_uid)
                                max_east = True
                                # print("max_east_level {}, consider_uid {}, considered_east_west_uid_list {}, max_east {}".format(max_east_level, considered_uid, considered_east_west_uid_list, max_east))

                            considered_south_north_uid_list = []
                            max_south = True
                            if "north" in reversed_considered_orients or "north_west" in reversed_considered_orients \
                                    or "north_east" in reversed_considered_orients:  # Tây Bắc, Bắc, Đông Bắc thì ưu tiên chọn điểm đỗ ở cực Nam
                                max_south_level = \
                                pspace_dict[max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["south_level"])][
                                    "south_level"]
                                considered_uid = list(dict(filter(lambda x: x[1]["south_level"] == max_south_level,
                                                                  pspace_dict.items())).keys())
                                considered_south_north_uid_list.extend(considered_uid)
                                # print("max_south_level {}, considered_uid {}, considered_south_north_uid_list{}, max_south {}".format(max_south_level, considered_uid, considered_south_north_uid_list, max_south))

                            elif "south" in reversed_considered_orients or "south_west" in reversed_considered_orients \
                                    or "south_east" in reversed_considered_orients:  # Tây Nam, Nam, Đông Nam thì ưu tiên chọn điểm đỗ ở cực Bắc
                                min_south_level = \
                                pspace_dict[min(pspace_dict.keys(), key=lambda x: pspace_dict[x]["south_level"])][
                                    "south_level"]
                                considered_uid = list(dict(filter(lambda x: x[1]["south_level"] == min_south_level,
                                                                  pspace_dict.items())).keys())
                                considered_south_north_uid_list.extend(considered_uid)
                                max_south = False
                                # print("min_south_level {}, considered_uid {}, considered_south_north_uid_list{}, max_south {}".format(min_south_level, considered_uid, considered_south_north_uid_list, max_south))

                            considered_uid = list(set(considered_east_west_uid_list).intersection(
                                considered_south_north_uid_list))  # Gộp các trường hợp trên trên lại

                            if len(considered_uid) == 0:  # Nếu hợp của hai trường hợp trên là rỗng
                                if len(considered_east_west_uid_list) > 0 or len(
                                        considered_south_north_uid_list) > 0:  # Parking space does not belong to any reversed considered orients, Nếu 1 trong hai trường hợp không rỗng
                                    chosen_uid = max(pspace_dict.keys(), key=lambda x: pspace_dict[x][
                                        "ios"])  # Chọn unified_id ứng với ios lớn nhất là "filled"
                                    unified_id_status_dict[chosen_uid] = "filled"
                                    uid_veh_id_match_list.append(
                                        [chosen_uid, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id,
                                         unified_id_to_ps[chosen_uid].type_space, self.parking_ground, cam])
                                    matched_vehicles_id_list.append(vehicle_id)
                                    for uid_match in pspace_dict:
                                        if uid_match != chosen_uid:
                                            if pspace_dict[uid_match][
                                                "ios"] > 0.75:  # Các unified_id còn lại cái nào ios lớn hơn 0.75 đặt là "unknown" ngược lại là "available
                                                unified_id_status_dict[uid_match] = "unknown"
                                            # else: # Nên sửa thành từ chuyển sang available thành giữ nguyên trạng thái vì rất có thể trạng thái đang filled
                                            #    unified_id_status_dict[uid_match] = "available"
                                else:  # Nếu cả hai trường hợp trên đều rỗng: Cứ unified_id nào có ios trên 0.65 thì trạng thái là "filled"
                                    for uid_match in pspace_dict:
                                        if pspace_dict[uid_match]["ios"] > 0.65:
                                            unified_id_status_dict[uid_match] = "filled"
                                            uid_veh_id_match_list.append(
                                                [uid_match, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id,
                                                 unified_id_to_ps[uid_match].type_space, self.parking_ground, cam])
                                            matched_vehicles_id_list.append(vehicle_id)
                            else:  # Nếu hợp của hai trường hợp trên không rỗng
                                try:
                                    assert len(considered_uid) == 1  # Xác nhận chỉ có một điểm đỗ (bug ở chỗ này
                                    considered_uid = considered_uid[0]
                                    filled_list = []
                                    unified_id_status_dict[
                                        considered_uid] = "filled"  # Điểm đỗ này được chọn là "filled"
                                    uid_veh_id_match_list.append(
                                        [considered_uid, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id,
                                         unified_id_to_ps[considered_uid].type_space, self.parking_ground, cam])
                                    matched_vehicles_id_list.append(vehicle_id)
                                    filled_list.append(
                                        considered_uid)  # Khởi tạo filled_list = []. filled_list thêm điểm đỗ vừa rồi
                                    if max_south:  # Nếu là max_south (ưu tiên south_level cao nhất)
                                        if "northern_adjacency" in pspace_dict[considered_uid][
                                            "adjacencies"]:  # Nếu điểm đỗ trên có lân cận phía Bắc và ios > 0.6 và loại xe là xe tải thì điểm đỗ lân cận này cũng được điền là "filled"
                                            north_of_considered_uid = pspace_dict[considered_uid]["adjacencies"][
                                                "northern_adjacency"]
                                            if pspace_dict[north_of_considered_uid][
                                                "ios"] > 0.6:  # and vehicles_list[col].class_id == 1 # "truck"
                                                unified_id_status_dict[north_of_considered_uid] = "filled"
                                                uid_veh_id_match_list.append([north_of_considered_uid, vehicle_id,
                                                                              vehicle_id_to_vehicle[
                                                                                  vehicle_id].class_id,
                                                                              unified_id_to_ps[
                                                                                  north_of_considered_uid].type_space,
                                                                              self.parking_ground, cam])
                                                matched_vehicles_id_list.append(vehicle_id)
                                                filled_list.append(north_of_considered_uid)
                                    else:  # Nếu là min_south (ưu tiên south_level thấp nhất)
                                        if "southern_adjacency" in pspace_dict[considered_uid][
                                            "adjacencies"]:  # Nếu điểm đỗ trên có lần cận phía Nam và ios > 0.6 và loại xe là xe tải thì điểm đỗ lân cận này cũng được điền là "filled"
                                            south_of_considered_uid = pspace_dict[considered_uid]["adjacencies"][
                                                "southern_adjacency"]
                                            if pspace_dict[south_of_considered_uid][
                                                "ios"] > 0.6:  # and vehicles_list[col].class_id == 1 # "truck"
                                                unified_id_status_dict[south_of_considered_uid] = "filled"
                                                uid_veh_id_match_list.append([south_of_considered_uid, vehicle_id,
                                                                              vehicle_id_to_vehicle[
                                                                                  vehicle_id].class_id,
                                                                              unified_id_to_ps[
                                                                                  south_of_considered_uid].type_space,
                                                                              self.parking_ground, cam])
                                                matched_vehicles_id_list.append(vehicle_id)
                                                filled_list.append(
                                                    south_of_considered_uid)  # filled_list thêm điểm trên vào
                                    for uid_match in pspace_dict:  # Xét các điểm đỗ còn lại (not in filled_list):
                                        if uid_match not in filled_list:
                                            if pspace_dict[uid_match][
                                                "ios"] > 0.7:  # Nếu ios > 0.7 thì điểm đỗ này được điền là "unknown
                                                unified_id_status_dict[uid_match] = "unknown"
                                            # else: # Nếu không thì điểm đỗ này là "available" # Nên sửa thành từ chuyển sang available thành giữ nguyên trạng thái vì rất có thể trạng thái đang filled
                                            #    unified_id_status_dict[uid_match] = "available"
                                except:
                                    print("consider_uid len is greater than one {}".format(considered_uid))
                                    print("Reversed_considered_orients {}".format(reversed_considered_orients))
                                    if not max_east:
                                        print(
                                            "min_east_level {}, consider_uid {}, considered_east_west_uid_list {}, max_east {}".format(
                                                min_east_level, considered_uid, considered_east_west_uid_list,
                                                max_east))
                                    else:
                                        print(
                                            "max_east_level {}, consider_uid {}, considered_east_west_uid_list {}, max_east {}".format(
                                                max_east_level, considered_uid, considered_east_west_uid_list,
                                                max_east))
                                    if max_south:
                                        print(
                                            "max_south_level {}, considered_uid {}, considered_south_north_uid_list{}, max_south {}".format(
                                                max_south_level, considered_uid, considered_south_north_uid_list,
                                                max_south))
                                    else:
                                        print(
                                            "min_south_level {}, considered_uid {}, considered_south_north_uid_list{}, max_south {}".format(
                                                min_south_level, considered_uid, considered_south_north_uid_list,
                                                max_south))
                                    print("Unified id {}, vehicle id {}, Pspace_dict {}".format(unified_id, vehicle_id,
                                                                                                pspace_dict))
                                    for uid_match in pspace_dict:
                                        if pspace_dict[uid_match]["ios"] > 0.5:
                                            unified_id_status_dict[uid_match] = "filled"
                                            uid_veh_id_match_list.append(
                                                [uid_match, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id,
                                                 unified_id_to_ps[uid_match].type_space, self.parking_ground, cam])
                                            matched_vehicles_id_list.append(vehicle_id)

                            # print("Unified id {}, vehicle id {}, Pspace_dict {}".format(unified_id, vehicle_id, pspace_dict))

                        considered_vehicle_id_list.append(vehicle_id)  # considered_vehicle_id_list thêm vehicle_id
        # print("Unified id status: {}".format(unified_id_status_dict))
        results_json[cam][frame_id]["matched_vehicles_id_list"] = matched_vehicles_id_list
        end = time.time()
        # print("This block consumes {} seconds".format(end - start))
        start = time.time()
        remained_vehicles_id_list = list(set(vehicle_id_to_vehicle.keys()) - set(matched_vehicles_id_list))
        # print("remained_vehicles_id_list: {}, vehicle square masks: {}".format(remained_vehicles_id_list, vehicle_square_of_mask.keys()))
        outlier_parking_spaces_in_cam_mask = self.parking_space_initializer.outlier_positions_mask[cam]
        # Tạo hai dictionary unified_id_to_vehicle_id_iov và vehicle_id_to_unified_id_iov chứa thông tin ios (intersection over vehicle) {unified_id1: {vehicle_id1: ..., vehicle_id2: ..., ...}, unified_id2: ...}, {vehicle_id1: {unified_id1:..., unified_id2:...,...}, vehicle_id2:...}
        outlier_unified_id_to_remained_vehicle_id_iov = {}
        remained_vehicle_id_to_outlier_unified_id_iov = {}
        if len(outlier_parking_spaces_in_cam) < len(
                remained_vehicles_id_list):  # Giả sử chọn vòng lặp for theo các key của dictionary của unified_id sang instance của parking_space:
            for unified_id in outlier_unified_id_to_ps:
                x_min, y_min, x_max, y_max = outlier_unified_id_to_ps[unified_id].bbox[
                    cam]  # Vì bbox là dict nên phải key cam vào một unified ứng với mỗi cam sẽ là một bbox khác nhau
                cropped_ps_mask = outlier_parking_spaces_in_cam_mask[y_min:y_max + 1,
                                  x_min:x_max + 1]  # Crop parking space positions mask
                cropped_vh_mask = vehicle_masks[y_min:y_max + 1, x_min:x_max + 1]  # Crop vehicle positions mask
                cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)
                # print("unified_id", unified_id, cropped_mask.shape)
                inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=unified_id,
                                                              use_unified_id=True)
                for ps_veh in inter_dict:
                    uid, vid = ps_veh
                    if vid not in remained_vehicles_id_list:
                        continue
                    inter = inter_dict[ps_veh]
                    iov = inter / vehicle_square_of_mask[vid]
                    if iov < iov_threshold:  # Tìm số lượng giao của unified_id với các vehicle_id trong vùng crop trên và lưu vào 2 dictionary unified_id_to_vehicle_id và vehicle_id_to_unified_id nếu ios thỏa mãn > threshold đặt trước, nếu không thì bỏ qua
                        continue
                    else:
                        if uid not in outlier_unified_id_to_remained_vehicle_id_iov:
                            outlier_unified_id_to_remained_vehicle_id_iov[uid] = {}
                        if vid not in outlier_unified_id_to_remained_vehicle_id_iov[uid]:
                            outlier_unified_id_to_remained_vehicle_id_iov[uid][vid] = inter / vehicle_square_of_mask[
                                vid]
                        else:
                            assert outlier_unified_id_to_remained_vehicle_id_iov[uid][vid] == (
                                    inter / vehicle_square_of_mask[vid]), "iov of 2 times is not equal"
                        if vid not in remained_vehicle_id_to_outlier_unified_id_iov:
                            remained_vehicle_id_to_outlier_unified_id_iov[vid] = {}
                        if uid not in remained_vehicle_id_to_outlier_unified_id_iov[vid]:
                            remained_vehicle_id_to_outlier_unified_id_iov[vid][uid] = inter / vehicle_square_of_mask[
                                vid]
                        else:
                            assert remained_vehicle_id_to_outlier_unified_id_iov[vid][uid] == (
                                        inter / vehicle_square_of_mask[vid]), "iov of 2 times is not equal"
        else:
            for vehicle_id in vehicle_id_to_vehicle:
                if vehicle_id in remained_vehicles_id_list:
                    x_min, y_min, x_max, y_max = vehicle_id_to_vehicle[vehicle_id].bbox
                    cropped_ps_mask = outlier_parking_spaces_in_cam_mask[y_min:y_max + 1,
                                      x_min:x_max + 1]  # Crop parking space positions mask
                    cropped_vh_mask = vehicle_masks[y_min:y_max + 1, x_min:x_max + 1]  # Crop vehicle positions mask
                    cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)
                    # print("vehicle_id", vehicle_id, cropped_mask.shape)
                    inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=vehicle_id,
                                                                  use_unified_id=False)
                    for ps_veh in inter_dict:
                        uid, vid = ps_veh
                        inter = inter_dict[ps_veh]
                        iov = inter / vehicle_square_of_mask[vid]
                        if iov < iov_threshold:
                            continue
                        else:
                            if uid not in outlier_unified_id_to_remained_vehicle_id_iov:
                                outlier_unified_id_to_remained_vehicle_id_iov[uid] = {}
                            if vid not in outlier_unified_id_to_remained_vehicle_id_iov[uid]:
                                outlier_unified_id_to_remained_vehicle_id_iov[uid][vid] = inter / \
                                                                                          vehicle_square_of_mask[vid]
                            else:
                                assert outlier_unified_id_to_remained_vehicle_id_iov[uid][vid] == (
                                            inter / vehicle_square_of_mask[vid]), "iov of 2 times is not equal"
                            if vid not in remained_vehicle_id_to_outlier_unified_id_iov:
                                remained_vehicle_id_to_outlier_unified_id_iov[vid] = {}
                            if uid not in remained_vehicle_id_to_outlier_unified_id_iov[vid]:
                                remained_vehicle_id_to_outlier_unified_id_iov[vid][uid] = inter / \
                                                                                          vehicle_square_of_mask[vid]
                            else:
                                assert remained_vehicle_id_to_outlier_unified_id_iov[vid][uid] == (
                                            inter / vehicle_square_of_mask[vid]), "iov of 2 times is not equal"
        # print("outlier_unified_id_to_remained_vehicle_id_iov: {}, remained_vehicle_id_to_outlier_unified_id_iov: {}".format(outlier_unified_id_to_remained_vehicle_id_iov, remained_vehicle_id_to_outlier_unified_id_iov))
        for unified_id in outlier_unified_id_to_remained_vehicle_id_iov:
            for vehicle_id in outlier_unified_id_to_remained_vehicle_id_iov[unified_id]:
                uid_veh_id_match_list.append([unified_id, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id,
                                              outlier_unified_id_to_ps[unified_id].type_space, self.parking_ground,
                                              cam])
                matched_vehicles_id_list.append(vehicle_id)
        matched_vehicles_id_list = list(set(matched_vehicles_id_list))
        vehicles_list = list(map(lambda x: vehicle_id_to_vehicle[x], matched_vehicles_id_list))
        # print([x.detection_id for x in vehicles_list], end="    ")
        if is_tracking:  # Nếu có sử dụng tracking
            assert tracker, "vehicles tracker cannot be None"
            tracker.step(vehicle_detections=vehicles_list)
            vehicles_list = tracker.get_result()  # Sử dụng track lấy ra các vehicle track là list các instance của vehicle_track
        # print(list(vehicle_id_to_vehicle.keys()), matched_vehicles_id_list, tracker.det_id_to_track_id_match_dict)
        results_json[cam][frame_id]["uid_veh_id_match_list_det_id"] = uid_veh_id_match_list
        if len(tracker.det_id_to_track_id_match_dict) > 0:
            uid_veh_id_match_list_copy = uid_veh_id_match_list.copy()
            uid_veh_id_match_list.clear()
            for matched_pair in uid_veh_id_match_list_copy:
                # print("match pair:", matched_pair, end="  ")
                if matched_pair[1] in tracker.det_id_to_track_id_match_dict:
                    matched_pair[1] = tracker.det_id_to_track_id_match_dict[matched_pair[1]]
                    uid_veh_id_match_list.append(tuple(matched_pair))
                # print("match pair:", matched_pair)
        else:
            uid_veh_id_match_list = []
        vehicle_id_to_vehicle = dict(map(lambda x: (x.track_id, x), vehicles_list))
        results_json[cam][frame_id]["uid_veh_id_match_list_track_id"] = uid_veh_id_match_list
        results_json[cam][frame_id]["det_id_to_track_id"] = tracker.det_id_to_track_id_match_dict
        results_json[cam][frame_id]["tracks_list"] = {}
        for vehicle in vehicles_list:
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id] = {}
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id]["score"] = vehicle.score
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id]["bbox"] = vehicle.bbox
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id]["positions"] = vehicle.positions
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id]["class_id"] = vehicle.class_id
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id]["track_id"] = vehicle.track_id
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id]["inactive_steps"] = vehicle.inactive_steps
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id]["parking_ground"] = vehicle.parking_ground
            results_json[cam][frame_id]["tracks_list"][vehicle.track_id]["cam"] = vehicle.cam

        end = time.time()
        # print("This block consumes {} seconds".format(end - start))
        start = time.time()
        # Visualize ảnh sử dụng các mask
        status_color_dict = {"filled": (0, 0, 255), "unknown": (0, 255, 255), "available": (0, 255, 0)}
        unified_id_to_color = {k: status_color_dict[v] for k, v in unified_id_status_dict.items()}
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        for uid, color in unified_id_to_color.items():
            color_mask[parking_spaces_in_cam_mask == uid] = color
        color_mask[vehicle_masks >= 0] = (255, 0, 0)
        color_mask[outlier_parking_spaces_in_cam_mask >= 0] = (255, 0, 255)

        frame = np.where(color_mask > 0, cv2.addWeighted(frame, 0.4, color_mask, 0.6, 0), frame)
        num_available_ps = len(
            list(filter(lambda x: unified_id_status_dict[x] == "available", unified_id_to_ps.keys())))
        num_vehicles = len(vehicle_id_to_vehicle.keys())
        for vehicle_id in vehicle_id_to_vehicle:
            x_min, y_min, x_max, y_max = vehicle_id_to_vehicle[vehicle_id].bbox
            track_id = vehicle_id_to_vehicle[vehicle_id].track_id
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), thickness=1)
            cv2.putText(frame, str(track_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0),
                        thickness=1)
        write_information(frame=frame, num_vehicles=num_vehicles, num_available_ps=num_available_ps)
        end = time.time()
        # print("This block consumes {} seconds".format(end - start))
        return unified_id_to_ps, vehicle_id_to_vehicle, unified_id_status_dict, frame, uid_veh_id_match_list

    def video_match(self, video_source_list, is_savevideo=False, save_dir=None, cam_list=["cam_1"], ios_threshold=0.3,
                    iov_threshold=0.4, is_tracking=True, is_showframe=True, tentative_steps_before_accepted=3,
                    tracking_tentative_steps_before_accepted=3, tracking_inactive_steps_before_removed=10,
                    pair_inactive_steps_before_removed=10, json_save_dir=None, results_json_name="results_json.json"):
        tracker_dict = {}
        if is_tracking:
            for cam in cam_list:
                tracker_dict[cam] = VehicleTracker(detection_vehicle_thresh=0.2,
                                                   inactive_steps_before_removed=tracking_inactive_steps_before_removed,
                                                   reid_iou_threshold=0.3,
                                                   max_traject_steps=50,
                                                   parking_ground=self.parking_ground,
                                                   tentative_steps_before_accepted=tracking_tentative_steps_before_accepted,
                                                   cam=cam)
        else:
            for cam in cam_list:
                tracker_dict[cam] = None

        if video_source_list[0].endswith((".mp4", ".avi")):
            start_time = get_start_time_from_video_name(source=video_source_list[0])
            use_time_stamp = False
        else:
            start_time = datetime.now()
            use_time_stamp = True

        pair_scheduler = JsonPairsScheduler(time=start_time, use_time_stamp=use_time_stamp, active_cams=cam_list,
                                            tentative_steps_before_accepted=tentative_steps_before_accepted,
                                            inactive_steps_before_removed=pair_inactive_steps_before_removed)

        output = {}
        for video_source, cam in zip(video_source_list, cam_list):
            cap = cv2.VideoCapture(video_source)

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if is_savevideo:
                assert save_dir, "When save video, save_dir cannot be None"
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                if video_source.endswith((".mp4", ".avi")):
                    video_name = os.path.basename(video_source)
                else:
                    video_name = "save_webcam.mp4"

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                output[cam] = cv2.VideoWriter(os.path.join(save_dir, video_name), fourcc, fps, (width, height))

            cap.release()

        stream_dict = {}

        for video_source, cam in zip(video_source_list, cam_list):

            stream_dict[cam] = QueuedStream(video_source, cam)
            stream_dict[cam].start()

            if not stream_dict[cam].isOpened():
                print("Can not open video: {}".format(video_source))
                self.detector.stop()
                raise StopIteration

        results_json = {}

        for cam in cam_list:
            results_json[cam] = {}

        self.run = True

        while self.run:

            for cam in cam_list:
                ret, frame, frame_id, time_stamp, cam_stream = stream_dict[cam].read()

                results_json[cam][frame_id] = {}

                if not ret:
                    self.run = False
                    break

                assert cam == cam_stream, "Cam of stream must be cam of this loop, cam {}, cam_stream {}".format(cam,
                                                                                                                 cam_stream)

                if self.run_multiprocessing:

                    self.detector.put_frame(frame_id, frame, time_stamp, cam_stream)

                    rois, scores, class_ids, masks, frame_id, time_stamp, cam_detect = self.detector.get_result()

                    self.positions_mask[cam_detect] = -1 * np.ones(shape=[height, width], dtype=np.int16)
                    self.square_of_mask[cam_detect] = OrderedDict()

                    detections_list = []

                    if self.model_arch == "mask_rcnn":
                        class_id_list = [1, 2, 3, 4]
                    else:
                        class_id_list = [2, 5, 7]
                    for det_id, (roi, score, class_id, mask) in enumerate(zip(rois, scores, class_ids, masks)):
                        if score >= self.detection_vehicle_thresh and class_id in class_id_list:
                            rr, cc = np.where(mask)
                            if len(rr) == 0 or len(cc) == 0:
                                continue
                            self.positions_mask[cam_detect][rr, cc] = det_id
                            self.square_of_mask[cam_detect][det_id] = rr.shape[0]
                            y_min, y_max = np.min(rr), np.max(rr)
                            x_min, x_max = np.min(cc), np.max(cc)
                            bbox = [x_min, y_min, x_max, y_max]
                            positions = np.array(
                                [rr,
                                 cc])  # Tập hợp các điểm [y1, y2, ..., yn], [x1, x2, ..., xn] nằm trong vehicle mask
                            if self.parking_ground == "parking_ground_SA" and cam_detect == "cam_1":  # Thêm điều kiện nếu là sân đỗ SA và camera là camera 1 thêm điều kiện để vùng nằm trên đường thẳng 9x + 10y - 5760 (góc trên bên trái màn hình), các xe được phát hiện trong vùng này sẽ bị bỏ qua
                                if 81 * x_max + 96 * y_max - 62208 >= 0:
                                    detections_list.append(
                                        VehicleDetection(score, bbox, positions, class_id, det_id, self.parking_ground,
                                                         cam_detect))
                                else:
                                    self.positions_mask[cam_detect][rr, cc] = -1
                            else:
                                detections_list.append(
                                    VehicleDetection(score, bbox, positions, class_id, det_id, self.parking_ground,
                                                     cam_detect))

                else:
                    cam_detect = cam_stream

                    detections_list = self.detector(frame, parking_ground=self.parking_ground, cam=cam_detect)

                    self.positions_mask[cam_detect] = self.detector.positions_mask[cam_detect]
                    self.square_of_mask[cam_detect] = self.detector.square_of_mask[cam_detect]

                results_json[cam][frame_id]["detections_list"] = {}
                for detection in detections_list:
                    results_json[cam][frame_id]["detections_list"][detection.detection_id] = {}
                    results_json[cam][frame_id]["detections_list"][detection.detection_id]["score"] = detection.score
                    results_json[cam][frame_id]["detections_list"][detection.detection_id]["bbox"] = detection.bbox
                    results_json[cam][frame_id]["detections_list"][detection.detection_id]["positions"] = detection.positions
                    results_json[cam][frame_id]["detections_list"][detection.detection_id]["class_id"] = detection.class_id
                    results_json[cam][frame_id]["detections_list"][detection.detection_id]["detection_id"] = detection.detection_id
                    results_json[cam][frame_id]["detections_list"][detection.detection_id]["parking_ground"] = detection.parking_ground
                    results_json[cam][frame_id]["detections_list"][detection.detection_id]["cam"] = detection.cam

                unified_id_to_ps, vehicle_id_to_vehicle, unified_id_status_dict, frame, uid_veh_id_match_list = self.frame_match(
                    frame=frame,
                    vehicles_list=detections_list,
                    results_json=results_json,
                    frame_id=frame_id,
                    cam=cam_detect,
                    ios_threshold=ios_threshold,
                    iov_threshold=iov_threshold,
                    is_tracking=is_tracking,
                    tracker=tracker_dict[cam_detect])
                pair_scheduler.step(uid_veh_list=uid_veh_id_match_list, num_frames=frame_id, time_stamp=time_stamp,
                                    cam=cam_detect, results_json=results_json, frame_stride=1, fps=fps)
                pair_scheduler.verify(cam=cam_detect)
                pairs = pair_scheduler.get_pairs_instances(cam=cam_detect)
                # print(vehicle_id_to_vehicle.keys())
                # print(uid_veh_id_match_list)
                # for pair in pairs:
                #    print("pair:", pair)
                # for uid, uid_pairs in groupby(dict(sorted(pairs.items(), key=lambda y: y[1].unified_id)).items(), key=lambda x: x[1].unified_id):
                #    #for uid_pair in uid_pairs:
                #    print(uid, len(list(uid_pairs)))
                if is_savevideo:
                    output[cam_detect].write(frame)
                if is_showframe:
                    cv2.imshow(cam_detect, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.run = False
                    if self.run_multiprocessing:
                        self.detector.stop()
                    for cam in cam_list:
                        stream_dict[cam].stop()
                    cv2.destroyAllWindows()
                    break

        if not os.path.exists(os.path.join(ROOT_DIR, json_save_dir)):
            os.makedirs(os.path.join(ROOT_DIR, json_save_dir), exist_ok=True)
        print(results_json)
        with open(os.path.join(ROOT_DIR, json_save_dir, results_json_name), "w") as f:
            json.dump(results_json, f, cls=NumpyEncoder)

        for cam in cam_list:
            pair_scheduler.save_pairs_to_db(cam=cam)
        if is_savevideo:
            for cam in cam_list:
                output[cam].release()
            print("Save video")
        print("Done")


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_source_list", type=str, default=r"G:\04_前沢SA_上\201911\20191130\カメラ1\2019-11-30_10-00-00.mp4", help="Path to video")
    parser.add_argument("--video_output_dir", type=str, default=r"F:\\", help="Path to output video")
    parser.add_argument("--json_output_dir", type=str, default="deploy_serving/jsons", help="Directory contains position mask and square mask of parking spaces")
    parser.add_argument("--results_json_name", type=str, default="results_json.json", help="Name of result json file")
    parser.add_argument("--model_arch", type=str, choices=["mask_rcnn", "yolact"], default="mask_rcnn", help="Model detector")
    parser.add_argument("--checkpoint_name", type=str, default="mask_rcnn_cars_and_vehicles_0008.h5", help="Checkpoint name")
    parser.add_argument("--detection_vehicle_thresh", type=float, default=0.4, help="Detection threshold")
    parser.add_argument("--cuda", type=bool, default=False, help="Use cuda or not")
    parser.add_argument("--ios_threshold", type=float, default=0.1, help="Threshold of ios to considered possibility of matching")
    parser.add_argument("--iov_threshold", type=float, default=0.1, help="Threshold of iov to considered posibility of matching")
    parser.add_argument("--is_showframe", type=str2bool, nargs="?", const=True, default=True, help="Show result or not")
    parser.add_argument("--parking_ground", type=str, default="parking_ground_SA", help="Parking ground")
    parser.add_argument("--cam_list", type=str, default="cam_1", help="Cam")
    parser.add_argument("--run_multiprocessing", type=str2bool, nargs="?", const=True, default=True, help="Run multiprocessing or not")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    video_source_list = [video for video in args.video_source_list.split(",")]
    cam_list = [cam for cam in args.cam_list.split(",")]
    assert len(video_source_list) == len(cam_list)
    matcher = JsonMatcher(active_cams=cam_list, parking_ground=args.parking_ground, model_arch=args.model_arch,
                          checkpoint_name=args.checkpoint_name, detection_vehicle_thresh=args.detection_vehicle_thresh,
                          run_multiprocessing=args.run_multiprocessing)
    matcher.video_match(video_source_list=video_source_list, is_savevideo=True, save_dir=args.video_output_dir,
                        cam_list=cam_list, ios_threshold=args.ios_threshold, iov_threshold=args.iov_threshold,
                        is_tracking=True, is_showframe=args.is_showframe, json_save_dir=args.json_output_dir,
                        results_json_name=args.results_json_name)
