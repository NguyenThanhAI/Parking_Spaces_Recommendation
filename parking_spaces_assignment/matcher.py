import sys
import os
from itertools import groupby
from tqdm import tqdm
import numpy as np
import cv2
from vehicle_tracking.vehicle_detector import VehicleDetector
from parking_spaces_assignment.parking_space import ParkingSpacesInitializer
from vehicle_tracking.vehicle_tracker import VehicleTracker
from parking_spaces_assignment.utils import find_unique_values_and_frequency, write_information
import time
from datetime import datetime
from load_videos.videos_utils import get_time_amount_from_frames_number, get_start_time_from_video_name
from parking_spaces_assignment.pair import PairsScheduler
from code_timing_profiling.profiling import profile
from code_timing_profiling.timing import timethis

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)


class Matcher(object):
    def __init__(self,
                 checkpoint_name="mask_rcnn_cars_and_vehicles_0008.h5",
                 model_arch="mask_rcnn",
                 cuda=False,
                 parking_ground="parking_ground_SA",
                 active_cams=["cam_2"],
                 shape=(720, 1280),
                 config_json_path=os.path.join(ROOT_DIR, "parking_spaces_data/parking_spaces_unified_id_segmen_in_cameras.json"),
                 detection_vehicle_thresh=0.4):
        self.parking_ground = parking_ground
        self.active_cams = active_cams
        self.parking_space_initializer = ParkingSpacesInitializer(active_cams=self.active_cams,
                                                                  parking_ground=self.parking_ground,
                                                                  shape=shape,
                                                                  config_json_path=config_json_path)
        #self.detectors_list = []
        #for cam in self.active_cams:
        #    self.detectors_list.append(VehicleDetector(checkpoint_name=checkpoint_name,
        #                                               cam=cam))
        self.detector = VehicleDetector(checkpoint_name=checkpoint_name, detection_vehicle_thresh=detection_vehicle_thresh,
                                        model_arch=model_arch, cuda=cuda)
        self.parking_spaces_list, self.outlier_parking_spaces_list = self.parking_space_initializer.initialize_parking_spaces()

    #@profile
    def frame_match(self, frame, cam="cam_2", ios_threshold=0.1, iov_threshold=0.4, is_tracking=False, tracker=None):

        assert cam in self.active_cams, "{} must be in {} of Matcher".format(cam, self.active_cams)

        vehicles_list = self.detector(frame=frame, parking_ground=self.parking_ground, cam=cam) # Phát hiện vehicle detection dưới dạng list các instance vehicle_detection

        if is_tracking: # Nếu có sử dụng tracking
            assert tracker, "vehicles tracker cannot be None"
            tracker.step(vehicle_detections=vehicles_list)
            vehicles_list = tracker.get_result() #Sử dụng track lấy ra các vehicle track là list các instance của vehicle_track
        start = time.time()
        parking_spaces_in_cam = list(filter(lambda x: cam in list(x.positions.keys()), self.parking_spaces_list)) # Lọc ra các parking spaces có trong cam hiện tại dưới dạng list các instance parking_space
        outlier_parking_spaces_in_cam = list(filter(lambda x: cam in list(x.positions.keys()), self.outlier_parking_spaces_list)) # Lọc ra các outlier parking spaces có trong cam hiện tại dưới dạng list các instance parking space
        if not is_tracking: # Tạo ra dictionary map từ vehicle_id (detection_id nếu không track, track_id nếu sử dụng track), sang instance của vehicle_detection (nếu không sử dụng track) hoặc vehicle_track (nếu sử dụng track)
            vehicle_id_to_vehicle = dict(map(lambda x: (x.detection_id, x), vehicles_list))
        else:
            vehicle_id_to_vehicle = dict(map(lambda x: (x.track_id, x), vehicles_list))

        unified_id_to_ps = dict(map(lambda x: (x.unified_id, x), parking_spaces_in_cam)) # Tạo ra dictionary map từ unified_id sang instance của parking_space
        outlier_unified_id_to_ps = dict(map(lambda x: (x.unified_id, x), outlier_parking_spaces_in_cam)) # Tạo ra dictionary map từ unified_id sang instance của parking_space (outlier parking space)

        if is_tracking: # Lấy positions_mask của parking_space từ parkingspaceinitializer._positions_map[cam] và positions_mask của vehicle từ vehicle_detector.positions_mask[cam] (nếu không sử dụng track) hoặc vehicle_tracker.positions_mask (nếu sử dụng track
            vehicle_masks = tracker.positions_mask
            vehicle_square_of_mask = tracker.square_of_mask
        else:
            vehicle_masks = self.detector.positions_mask[cam]
            vehicle_square_of_mask = self.detector.square_of_mask[cam]
        parking_spaces_in_cam_mask = self.parking_space_initializer.positions_mask[cam]
        # Tạo hai dictionary unified_id_to_vehicle_id_ios và vehicle_id_to_unified_id_ios chứa thông tin ios (intersection over space) {unified_id1: {vehicle_id1: ..., vehicle_id2: ..., ...}, unified_id2: ...}, {vehicle_id1: {unified_id1:..., unified_id2:...,...}, vehicle_id2:...}
        unified_id_to_vehicle_id_ios = {}
        vehicle_id_to_unified_id_ios = {}

        if len(parking_spaces_in_cam) < len(vehicles_list): # Giả sử chọn vòng lặp for theo các key của dictionary của unified_id sang instance của parking_space:
            for unified_id in unified_id_to_ps:
                x_min, y_min, x_max, y_max = unified_id_to_ps[unified_id].bbox[cam] # Vì bbox là dict nên phải key cam vào một unified ứng với mỗi cam sẽ là một bbox khác nhau
                cropped_ps_mask = parking_spaces_in_cam_mask[y_min:y_max + 1, x_min:x_max + 1] # Crop parking space positions mask
                cropped_vh_mask = vehicle_masks[y_min:y_max + 1, x_min:x_max + 1] # Crop vehicle positions mask
                cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)
                #print("unified_id", unified_id, cropped_mask.shape)
                inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=unified_id, use_unified_id=True)
                for ps_veh in inter_dict:
                    uid, vid = ps_veh
                    inter = inter_dict[ps_veh]
                    ios = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                    if ios < ios_threshold: # Tìm số lượng giao của unified_id với các vehicle_id trong vùng crop trên và lưu vào 2 dictionary unified_id_to_vehicle_id và vehicle_id_to_unified_id nếu ios thỏa mãn > threshold đặt trước, nếu không thì bỏ qua
                        continue
                    else:
                        if uid not in unified_id_to_vehicle_id_ios:
                            unified_id_to_vehicle_id_ios[uid] = {}
                        if vid not in unified_id_to_vehicle_id_ios[uid]:
                            unified_id_to_vehicle_id_ios[uid][vid] = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                        else:
                            assert unified_id_to_vehicle_id_ios[uid][vid] == (inter / self.parking_space_initializer.square_of_mask[cam][uid]), "ios of 2 times is not equal"
                        if vid not in vehicle_id_to_unified_id_ios:
                            vehicle_id_to_unified_id_ios[vid] = {}
                        if uid not in vehicle_id_to_unified_id_ios[vid]:
                            vehicle_id_to_unified_id_ios[vid][uid] = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                        else:
                            assert vehicle_id_to_unified_id_ios[vid][uid] == (inter / self.parking_space_initializer.square_of_mask[cam][uid]), "ios of 2 times is not equal"
        else:
            for vehicle_id in vehicle_id_to_vehicle:
                x_min, y_min, x_max, y_max = vehicle_id_to_vehicle[vehicle_id].bbox
                cropped_ps_mask = parking_spaces_in_cam_mask[y_min:y_max + 1, x_min:x_max + 1]  # Crop parking space positions mask
                cropped_vh_mask = vehicle_masks[y_min:y_max + 1, x_min:x_max + 1]  # Crop vehicle positions mask
                cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)
                #print("vehicle_id", vehicle_id, cropped_mask.shape)
                inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=vehicle_id, use_unified_id=False)
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
                            unified_id_to_vehicle_id_ios[uid][vid] = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                        else:
                            assert unified_id_to_vehicle_id_ios[uid][vid] == (inter / self.parking_space_initializer.square_of_mask[cam][uid]), "ios of 2 times is not equal"
                        if vid not in vehicle_id_to_unified_id_ios:
                            vehicle_id_to_unified_id_ios[vid] = {}
                        if uid not in vehicle_id_to_unified_id_ios[vid]:
                            vehicle_id_to_unified_id_ios[vid][uid] = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                        else:
                            assert vehicle_id_to_unified_id_ios[vid][uid] == (inter / self.parking_space_initializer.square_of_mask[cam][uid]), "ios of 2 times is not equal"
        #print("unified_id_to_vehicle_id_ios: {}, vehicle_id_to_unified_id_ios: {}".format(unified_id_to_vehicle_id_ios, vehicle_id_to_unified_id_ios))
        end = time.time()
        #print("This block consumes {} seconds".format(end - start))
        unified_id_status_dict = dict(zip(list(unified_id_to_ps.keys()), ["available"]*len(list(unified_id_to_ps.keys())))) # Tạo một unified_id_status_dict = {unified_id: "unknown", ....} tất cả các unified_id có trạng thái ban đầu là unknown # Từ filled thành unknown
        start = time.time()
        uid_veh_id_match_list = []
        matched_vehicles_id_list = []
        considered_vehicle_id_list = [] # Đặt một considered_vehicle_id_list = [] chứa các vehicle_id đã được xét với các unified_id
        for unified_id in unified_id_status_dict: # Duyệt từng unified_id trên unified_id_status_dict:
            if unified_id not in unified_id_to_vehicle_id_ios: # Nếu unified_id không là tồn tại là key trong unified_id_to_vehicle_id thì chuyển trạng thái của unified_id trong unified_id_status_dict là "available"
                unified_id_status_dict[unified_id] = "available"
            else: # unified_id là tồn tại là key trong unified_id_to_vehicle_id
                for vehicle_id in unified_id_to_vehicle_id_ios[unified_id]: # Xét từng vehicle_id có ios giao trong unified_id_to_vehicle_id[unified_id]
                    if vehicle_id not in considered_vehicle_id_list: # Nếu vehicle_id này không nằm trong considered_vehicle_id_list = [] thì xét tiếp, ngược lại đã ở trong rồi thì bỏ qua chuyển sang vehicle_id tiếp theo
                        assert unified_id in vehicle_id_to_unified_id_ios[vehicle_id], "Parking space id {} must be in vehicle to parking space ios {}".format(unified_id, vehicle_id)
                        if len(vehicle_id_to_unified_id_ios[vehicle_id]) == 1: # Nếu vehicle_id_to_unified_id[vehicle_id] của vehicle_id đang xét này chỉ có đúng một unified_id đang xét
                            unified_id_status_dict[unified_id] = "filled"
                            uid_veh_id_match_list.append((unified_id, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id, unified_id_to_ps[unified_id].type_space, self.parking_ground, cam))
                            matched_vehicles_id_list.append(vehicle_id)
                            #print("Parking space unified id {} and vehicle id {} is matched".format(unified_id, vehicle_id))
                        else: # Nếu vehicle_id_to_unified_id[vehicle_id] của vehicle_id đang xét nhiều hơn một unified_id
                            pspace_dict = {}  # Dictionary of dictionary, each dictionary represents and parking space with information east level, south level and adjacencies against orient
                            for uid_match in vehicle_id_to_unified_id_ios[vehicle_id]: # Xét từng unified_id này, tạo một pspace_dict lưu trữ các thông tin: south_level, east_level, visited, adjacencies, ios, reversed_considered_orients (tất nhiên phải tương ứng với cam)
                                uid_match_dict = {}
                                uid_match_dict["east_level"] = 0
                                uid_match_dict["south_level"] = 0
                                uid_match_dict["visited"] = False
                                pspace = unified_id_to_ps[uid_match]
                                adjacencies = dict(filter(lambda x: x[1] is not None, pspace.adjacencies.items()))
                                adjacencies = dict(filter(lambda x: x[1] in vehicle_id_to_unified_id_ios[vehicle_id], adjacencies.items()))
                                uid_match_dict["adjacencies"] = adjacencies
                                uid_match_dict["ios"] = vehicle_id_to_unified_id_ios[vehicle_id][uid_match]
                                uid_match_dict["reversed_considered_orients"] = pspace.reversed_considered_orients[cam] if cam in pspace.reversed_considered_orients else {}
                                pspace_dict[uid_match] = uid_match_dict

                            trace = []
                            def traverse_neighbors(uid_match): # Sử dụng đệ quy tương hỗ để tính south_level, east_level của từng unified_id
                                #pspace[uid_match]["visited"] = True
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

                            pspace_dict[random_uid]["visited"] = True # Choose random_uid as starting point
                            trace.append(random_uid) # Initialize track
                            traverse_neighbors(random_uid) # Assign value to east_level and south_level

                            #print("Random unified_id {}, trace {}".format(random_uid, trace))

                            reversed_considered_orients = {} # Tạo reversed_considered_orients = {"orients": [unified_id1, unified_id2, ...], ....}
                            for uid_match in pspace_dict:
                                orients = pspace_dict[uid_match]["reversed_considered_orients"]
                                for orient in orients:
                                    if orient not in reversed_considered_orients:
                                        reversed_considered_orients[orient] = []
                                    reversed_considered_orients[orient].append(uid_match)
                            #print("Reversed_considered_orients {}".format(reversed_considered_orients))
                            # Xét các hướng nếu trong reversed_considered_orients có
                            considered_east_west_uid_list = []
                            max_east = False
                            if "north_east" in reversed_considered_orients or "east" in reversed_considered_orients \
                                  or "south_east" in reversed_considered_orients: # Tây Nam, Tây Bắc, Tây thì ưu tiên chọn điểm đỗ ở cực Đông
                                min_east_level = pspace_dict[min(pspace_dict.keys(), key=lambda x: pspace_dict[x]["east_level"])]["east_level"]
                                considered_uid = list(dict(filter(lambda x: x[1]["east_level"] == min_east_level, pspace_dict.items())).keys())
                                considered_east_west_uid_list.extend(considered_uid)
                                #print("min_east_level {}, consider_uid {}, considered_east_west_uid_list {}, max_east {}".format(min_east_level, considered_uid, considered_east_west_uid_list, max_east))

                            elif "north_west"  in reversed_considered_orients or "west" in reversed_considered_orients \
                                    or "south_west" in reversed_considered_orients: # Đông Nam, Đông Bắc, Đông thì ưu tiên chọn điểm đỗ ở cực Tây
                                max_east_level = pspace_dict[max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["east_level"])]["east_level"]
                                considered_uid = list(dict(filter(lambda x: x[1]["east_level"] == max_east_level, pspace_dict.items())).keys())
                                considered_east_west_uid_list.extend(considered_uid)
                                max_east = True
                                #print("max_east_level {}, consider_uid {}, considered_east_west_uid_list {}, max_east {}".format(max_east_level, considered_uid, considered_east_west_uid_list, max_east))

                            considered_south_north_uid_list = []
                            max_south = True
                            if "north" in reversed_considered_orients or "north_west" in reversed_considered_orients \
                                    or "north_east" in reversed_considered_orients: # Tây Bắc, Bắc, Đông Bắc thì ưu tiên chọn điểm đỗ ở cực Nam
                                max_south_level = pspace_dict[max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["south_level"])]["south_level"]
                                considered_uid = list(dict(filter(lambda x: x[1]["south_level"] == max_south_level, pspace_dict.items())).keys())
                                considered_south_north_uid_list.extend(considered_uid)
                                #print("max_south_level {}, considered_uid {}, considered_south_north_uid_list{}, max_south {}".format(max_south_level, considered_uid, considered_south_north_uid_list, max_south))

                            elif "south" in reversed_considered_orients or "south_west" in reversed_considered_orients \
                                    or "south_east" in reversed_considered_orients: # Tây Nam, Nam, Đông Nam thì ưu tiên chọn điểm đỗ ở cực Bắc
                                min_south_level = pspace_dict[min(pspace_dict.keys(), key=lambda x: pspace_dict[x]["south_level"])]["south_level"]
                                considered_uid = list(dict(filter(lambda x: x[1]["south_level"] == min_south_level, pspace_dict.items())).keys())
                                considered_south_north_uid_list.extend(considered_uid)
                                max_south = False
                                #print("min_south_level {}, considered_uid {}, considered_south_north_uid_list{}, max_south {}".format(min_south_level, considered_uid, considered_south_north_uid_list, max_south))

                            considered_uid = list(set(considered_east_west_uid_list).intersection(considered_south_north_uid_list)) # Gộp các trường hợp trên trên lại

                            if len(considered_uid) == 0: # Nếu hợp của hai trường hợp trên là rỗng
                                if len(considered_east_west_uid_list) > 0 or len(considered_south_north_uid_list) > 0: # Parking space does not belong to any reversed considered orients, Nếu 1 trong hai trường hợp không rỗng
                                    chosen_uid = max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["ios"]) # Chọn unified_id ứng với ios lớn nhất là "filled"
                                    unified_id_status_dict[chosen_uid] = "filled"
                                    uid_veh_id_match_list.append((chosen_uid, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id, unified_id_to_ps[chosen_uid].type_space, self.parking_ground, cam))
                                    matched_vehicles_id_list.append(vehicle_id)
                                    for uid_match in pspace_dict:
                                        if uid_match != chosen_uid:
                                            if pspace_dict[uid_match]["ios"] > 0.75: # Các unified_id còn lại cái nào ios lớn hơn 0.75 đặt là "unknown" ngược lại là "available
                                                unified_id_status_dict[uid_match] = "unknown"
                                            #else: # Nên sửa thành từ chuyển sang available thành giữ nguyên trạng thái vì rất có thể trạng thái đang filled
                                            #    unified_id_status_dict[uid_match] = "available"
                                else: # Nếu cả hai trường hợp trên đều rỗng: Cứ unified_id nào có ios trên 0.65 thì trạng thái là "filled"
                                    for uid_match in pspace_dict:
                                        if pspace_dict[uid_match]["ios"] > 0.65:
                                            unified_id_status_dict[uid_match] = "filled"
                                            uid_veh_id_match_list.append((uid_match, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id, unified_id_to_ps[uid_match].type_space, self.parking_ground, cam))
                                            matched_vehicles_id_list.append(vehicle_id)
                            else: # Nếu hợp của hai trường hợp trên không rỗng
                                try:
                                    assert len(considered_uid) == 1 # Xác nhận chỉ có một điểm đỗ (bug ở chỗ này
                                    considered_uid = considered_uid[0]
                                    filled_list = []
                                    unified_id_status_dict[considered_uid] = "filled" # Điểm đỗ này được chọn là "filled"
                                    uid_veh_id_match_list.append((considered_uid, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id, unified_id_to_ps[considered_uid].type_space, self.parking_ground, cam))
                                    matched_vehicles_id_list.append(vehicle_id)
                                    filled_list.append(considered_uid) # Khởi tạo filled_list = []. filled_list thêm điểm đỗ vừa rồi
                                    if max_south: # Nếu là max_south (ưu tiên south_level cao nhất)
                                        if "northern_adjacency" in pspace_dict[considered_uid]["adjacencies"]: # Nếu điểm đỗ trên có lân cận phía Bắc và ios > 0.6 và loại xe là xe tải thì điểm đỗ lân cận này cũng được điền là "filled"
                                            north_of_considered_uid = pspace_dict[considered_uid]["adjacencies"]["northern_adjacency"]
                                            if pspace_dict[north_of_considered_uid]["ios"] > 0.6: # and vehicles_list[col].class_id == 1 # "truck"
                                                unified_id_status_dict[north_of_considered_uid] = "filled"
                                                uid_veh_id_match_list.append((north_of_considered_uid, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id, unified_id_to_ps[north_of_considered_uid].type_space, self.parking_ground, cam))
                                                matched_vehicles_id_list.append(vehicle_id)
                                                filled_list.append(north_of_considered_uid)
                                    else: # Nếu là min_south (ưu tiên south_level thấp nhất)
                                        if "southern_adjacency" in pspace_dict[considered_uid]["adjacencies"]: # Nếu điểm đỗ trên có lần cận phía Nam và ios > 0.6 và loại xe là xe tải thì điểm đỗ lân cận này cũng được điền là "filled"
                                            south_of_considered_uid = pspace_dict[considered_uid]["adjacencies"]["southern_adjacency"]
                                            if pspace_dict[south_of_considered_uid]["ios"] > 0.6: # and vehicles_list[col].class_id == 1 # "truck"
                                                unified_id_status_dict[south_of_considered_uid] = "filled"
                                                uid_veh_id_match_list.append((south_of_considered_uid, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id, unified_id_to_ps[south_of_considered_uid].type_space, self.parking_ground, cam))
                                                matched_vehicles_id_list.append(vehicle_id)
                                                filled_list.append(south_of_considered_uid) # filled_list thêm điểm trên vào
                                    for uid_match in pspace_dict: # Xét các điểm đỗ còn lại (not in filled_list):
                                        if uid_match not in filled_list:
                                            if pspace_dict[uid_match]["ios"] > 0.7: # Nếu ios > 0.7 thì điểm đỗ này được điền là "unknown
                                                unified_id_status_dict[uid_match] = "unknown"
                                            #else: # Nếu không thì điểm đỗ này là "available" # Nên sửa thành từ chuyển sang available thành giữ nguyên trạng thái vì rất có thể trạng thái đang filled
                                            #    unified_id_status_dict[uid_match] = "available"
                                except:
                                    print("consider_uid len is greater than one {}".format(considered_uid))
                                    print("Reversed_considered_orients {}".format(reversed_considered_orients))
                                    if not max_east:
                                        print("min_east_level {}, consider_uid {}, considered_east_west_uid_list {}, max_east {}".format(min_east_level, considered_uid, considered_east_west_uid_list, max_east))
                                    else:
                                        print("max_east_level {}, consider_uid {}, considered_east_west_uid_list {}, max_east {}".format(max_east_level, considered_uid, considered_east_west_uid_list, max_east))
                                    if max_south:
                                        print("max_south_level {}, considered_uid {}, considered_south_north_uid_list{}, max_south {}".format(max_south_level, considered_uid, considered_south_north_uid_list, max_south))
                                    else:
                                        print("min_south_level {}, considered_uid {}, considered_south_north_uid_list{}, max_south {}".format(min_south_level, considered_uid, considered_south_north_uid_list, max_south))
                                    print("Unified id {}, vehicle id {}, Pspace_dict {}".format(unified_id, vehicle_id, pspace_dict))
                                    for uid_match in pspace_dict:
                                        if pspace_dict[uid_match]["ios"] > 0.5:
                                            unified_id_status_dict[uid_match] = "filled"
                                            uid_veh_id_match_list.append((uid_match, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id, unified_id_to_ps[uid_match].type_space, self.parking_ground, cam))
                                            matched_vehicles_id_list.append(vehicle_id)

                            #print("Unified id {}, vehicle id {}, Pspace_dict {}".format(unified_id, vehicle_id, pspace_dict))

                        considered_vehicle_id_list.append(vehicle_id) # considered_vehicle_id_list thêm vehicle_id
        #print("Unified id status: {}".format(unified_id_status_dict))
        end = time.time()
        #print("This block consumes {} seconds".format(end - start))
        start = time.time()
        remained_vehicles_id_list = list(set(vehicle_id_to_vehicle.keys()) - set(matched_vehicles_id_list))
        #print("remained_vehicles_id_list: {}, vehicle square masks: {}".format(remained_vehicles_id_list, vehicle_square_of_mask.keys()))
        outlier_parking_spaces_in_cam_mask = self.parking_space_initializer.outlier_positions_mask[cam]
        # Tạo hai dictionary unified_id_to_vehicle_id_iov và vehicle_id_to_unified_id_iov chứa thông tin ios (intersection over vehicle) {unified_id1: {vehicle_id1: ..., vehicle_id2: ..., ...}, unified_id2: ...}, {vehicle_id1: {unified_id1:..., unified_id2:...,...}, vehicle_id2:...}
        outlier_unified_id_to_remained_vehicle_id_iov = {}
        remained_vehicle_id_to_outlier_unified_id_iov = {}
        if len(outlier_parking_spaces_in_cam) < len(remained_vehicles_id_list):  # Giả sử chọn vòng lặp for theo các key của dictionary của unified_id sang instance của parking_space:
            for unified_id in outlier_unified_id_to_ps:
                x_min, y_min, x_max, y_max = outlier_unified_id_to_ps[unified_id].bbox[cam]  # Vì bbox là dict nên phải key cam vào một unified ứng với mỗi cam sẽ là một bbox khác nhau
                cropped_ps_mask = outlier_parking_spaces_in_cam_mask[y_min:y_max + 1, x_min:x_max + 1]  # Crop parking space positions mask
                cropped_vh_mask = vehicle_masks[y_min:y_max + 1, x_min:x_max + 1]  # Crop vehicle positions mask
                cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)
                # print("unified_id", unified_id, cropped_mask.shape)
                inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=unified_id, use_unified_id=True)
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
                            outlier_unified_id_to_remained_vehicle_id_iov[uid][vid] = inter / vehicle_square_of_mask[vid]
                        else:
                            assert outlier_unified_id_to_remained_vehicle_id_iov[uid][vid] == (
                                        inter / vehicle_square_of_mask[vid]), "iov of 2 times is not equal"
                        if vid not in remained_vehicle_id_to_outlier_unified_id_iov:
                            remained_vehicle_id_to_outlier_unified_id_iov[vid] = {}
                        if uid not in remained_vehicle_id_to_outlier_unified_id_iov[vid]:
                            remained_vehicle_id_to_outlier_unified_id_iov[vid][uid] = inter / vehicle_square_of_mask[vid]
                        else:
                            assert remained_vehicle_id_to_outlier_unified_id_iov[vid][uid] == (inter / vehicle_square_of_mask[vid]), "iov of 2 times is not equal"
        else:
            for vehicle_id in vehicle_id_to_vehicle:
                if vehicle_id in remained_vehicles_id_list:
                    x_min, y_min, x_max, y_max = vehicle_id_to_vehicle[vehicle_id].bbox
                    cropped_ps_mask = outlier_parking_spaces_in_cam_mask[y_min:y_max + 1, x_min:x_max + 1]  # Crop parking space positions mask
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
                                outlier_unified_id_to_remained_vehicle_id_iov[uid][vid] = inter / vehicle_square_of_mask[vid]
                            else:
                                assert outlier_unified_id_to_remained_vehicle_id_iov[uid][vid] == (inter / vehicle_square_of_mask[vid]), "iov of 2 times is not equal"
                            if vid not in remained_vehicle_id_to_outlier_unified_id_iov:
                                remained_vehicle_id_to_outlier_unified_id_iov[vid] = {}
                            if uid not in remained_vehicle_id_to_outlier_unified_id_iov[vid]:
                                remained_vehicle_id_to_outlier_unified_id_iov[vid][uid] = inter / vehicle_square_of_mask[vid]
                            else:
                                assert remained_vehicle_id_to_outlier_unified_id_iov[vid][uid] == (inter / vehicle_square_of_mask[vid]), "iov of 2 times is not equal"
        #print("outlier_unified_id_to_remained_vehicle_id_iov: {}, remained_vehicle_id_to_outlier_unified_id_iov: {}".format(outlier_unified_id_to_remained_vehicle_id_iov, remained_vehicle_id_to_outlier_unified_id_iov))
        for unified_id in outlier_unified_id_to_remained_vehicle_id_iov:
            for vehicle_id in outlier_unified_id_to_remained_vehicle_id_iov[unified_id]:
                uid_veh_id_match_list.append((unified_id, vehicle_id, vehicle_id_to_vehicle[vehicle_id].class_id, outlier_unified_id_to_ps[unified_id].type_space, self.parking_ground, cam))
        end = time.time()
        #print("This block consumes {} seconds".format(end - start))
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
        num_available_ps = len(list(filter(lambda x: unified_id_status_dict[x] == "available", unified_id_to_ps.keys())))
        num_vehicles = len(vehicle_id_to_vehicle.keys())
        for vehicle_id in vehicle_id_to_vehicle:
            x_min, y_min, x_max, y_max = vehicle_id_to_vehicle[vehicle_id].bbox
            track_id = vehicle_id_to_vehicle[vehicle_id].track_id
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), thickness=1)
            cv2.putText(frame, str(track_id), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), thickness=1)
        write_information(frame=frame, num_vehicles=num_vehicles, num_available_ps=num_available_ps)
        end = time.time()
        #print("This block consumes {} seconds".format(end - start))
        return unified_id_to_ps, vehicle_id_to_vehicle, unified_id_status_dict, frame, uid_veh_id_match_list

    def image_match(self, image_path, save_dir, cam="cam_1", ios_threshold=0.3, iov_threshold=0.4, is_tracking=False, is_showimage=True):
        if is_tracking:
            tracker = VehicleTracker(detection_vehicle_thresh=0.2,
                                     inactive_steps_before_removed=10,
                                     reid_iou_threshold=0.3,
                                     max_traject_steps=50,
                                     parking_ground=self.parking_ground,
                                     cam=cam)
        else:
            tracker = None
        image = cv2.imread(image_path)
        unified_id_to_ps, vehicle_id_to_vehicle, unified_id_status_dict, frame, uid_veh_id_match_list = self.frame_match(frame=image, cam=cam, ios_threshold=ios_threshold, iov_threshold=iov_threshold, is_tracking=is_tracking, tracker=tracker)
        if is_showimage:
            cv2.imshow("", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        results_path = os.path.join(save_dir, os.path.basename(image_path).split(".")[0] + ".jpg")
        cv2.imwrite(results_path, frame)

    def video_match(self, video_source, is_savevideo=False, save_dir=None, cam="cam_1", ios_threshold=0.3, iov_threshold=0.4, is_tracking=True, is_showframe=True, tentative_steps_before_accepted=30, tracking_inactive_steps_before_removed=200, pair_inactive_steps_before_removed=1000):
        if is_tracking:
            tracker = VehicleTracker(detection_vehicle_thresh=0.2,
                                     inactive_steps_before_removed=tracking_inactive_steps_before_removed,
                                     reid_iou_threshold=0.3,
                                     max_traject_steps=50,
                                     parking_ground=self.parking_ground,
                                     cam=cam)
        else:
            tracker = None

        if video_source.endswith((".mp4", ".avi")):
            start_time = get_start_time_from_video_name(source=video_source)
        else:
            start_time = datetime.now()

        pair_scheduler = PairsScheduler(time=start_time, tentative_steps_before_accepted=tentative_steps_before_accepted, inactive_steps_before_removed=pair_inactive_steps_before_removed)

        cap = cv2.VideoCapture(video_source)

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if is_savevideo:
            assert save_dir, "When save video, save_dir cannot be None"
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            if video_source.endswith((".mp4", ".avi")):
                video_name = os.path.basename(video_source)
            else:
                video_name = "save_webcam.mp4"

            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            output = cv2.VideoWriter(os.path.join(save_dir, video_name), fourcc, fps, (width, height))

        stopped = False

        for i in tqdm(range(length)):
            if not stopped:
                ret, frame = cap.read()

                if not ret:
                    stopped = True
                    continue

                unified_id_to_ps, vehicle_id_to_vehicle, unified_id_status_dict, frame, uid_veh_id_match_list = self.frame_match(frame=frame,
                                                                                                                                 cam=cam,
                                                                                                                                 ios_threshold=ios_threshold,
                                                                                                                                 iov_threshold=iov_threshold,
                                                                                                                                 is_tracking=is_tracking,
                                                                                                                                 tracker=tracker)
                pair_scheduler.step(uid_veh_list=uid_veh_id_match_list, num_frames=i, frame_stride=1, fps=fps)
                pair_scheduler.verify()
                pairs = pair_scheduler.get_pairs_instances()
                #print(vehicle_id_to_vehicle.keys())
                #print(uid_veh_id_match_list)
                #for pair in pairs:
                #    print("pair:", pair)
                #for uid, uid_pairs in groupby(dict(sorted(pairs.items(), key=lambda y: y[1].unified_id)).items(), key=lambda x: x[1].unified_id):
                #    #for uid_pair in uid_pairs:
                #    print(uid, len(list(uid_pairs)))

                if is_savevideo:
                    output.write(frame)
                if is_showframe:
                    cv2.imshow("", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped = True
            else:
                cv2.destroyAllWindows()
                break

        pair_scheduler.save_pairs_to_db()
        if is_savevideo:
            output.release()
            print("Save video")
        print("Done")


#matcher = Matcher()
#tracker = VehicleTracker(detection_vehicle_thresh=0.2,
#                         inactive_steps_before_removed=10,
#                         reid_iou_threshold=0.3,
#                         max_traject_steps=50,
#                         parking_ground="parking_ground_SA",
#                         cam="cam_2")
#image_path = "../test_object_detection_models/images/201909_20190914_2_2019-09-14_05-00-00_8987.jpg"
#demo_images_dir = r"E:\demo_images"
#image = cv2.imread(image_path)
#matcher.image_match(image_path=image_path,
#                    save_dir=demo_images_dir,
#                    cam="cam_2",
#                    threshold=0.3,
#                    is_tracking=True,
#                    is_showimage=True)
