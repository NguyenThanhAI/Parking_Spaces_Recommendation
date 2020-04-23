import sys
import os
from tqdm import tqdm
import numpy as np
import cv2
from vehicle_tracking.vehicle_detector import VehicleDetector
from parking_spaces_assignment.parking_space import ParkingSpacesInitializer
from vehicle_tracking.vehicle_tracker import VehicleTracker
from parking_spaces_assignment.utils import find_unique_values_and_frequency
import time
from code_timing_profiling.profiling import profile
from code_timing_profiling.timing import timethis

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)


class Matcher(object):
    def __init__(self,
                 checkpoint_name="mask_rcnn_cars_and_vehicles_0008.h5",
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
        self.detector = VehicleDetector(checkpoint_name=checkpoint_name, detection_vehicle_thresh=detection_vehicle_thresh)
        self.parking_spaces_list = self.parking_space_initializer.initialize_parking_spaces()

    #@profile
    def frame_match(self, frame, cam="cam_2", threshold=0.3, is_tracking=False, tracker=None):

        assert cam in self.active_cams

        vehicles_list = self.detector(frame=frame, parking_ground=self.parking_ground, cam=cam) # Phát hiện vehicle detection dưới dạng list các instance vehicle_detection

        if is_tracking: # Nếu có sử dụng tracking
            assert tracker, "vehicles tracker cannot be None"
            tracker.step(vehicle_detections=vehicles_list)
            vehicles_list = tracker.get_result() #Sử dụng track lấy ra các vehicle track là list các instance của vehicle_track
        start = time.time()
        parking_spaces_in_cam = list(filter(lambda x: cam in list(x.positions.keys()), self.parking_spaces_list)) # Lọc ra các parking spaces có trong cam hiện tại dưới dạng list các instance parking_space
        if not is_tracking: # Tạo ra dictionary map từ vehicle_id (detection_id nếu không track, track_id nếu sử dụng track), sang instance của vehicle_detection (nếu không sử dụng track) hoặc vehicle_track (nếu sử dụng track)
            vehicle_id_to_vehicle = dict(map(lambda x: (x.detection_id, x), vehicles_list))
        else:
            vehicle_id_to_vehicle = dict(map(lambda x: (x.track_id, x), vehicles_list))

        unified_id_to_ps = dict(map(lambda x: (x.unified_id, x), parking_spaces_in_cam)) # Tạo ra dictionary map từ unified_id sang instance của parking_space

        if is_tracking: # Lấy positions_mask của parking_space từ parkingspaceinitializer._positions_map[cam] và positions_mask của vehicle từ vehicle_detector.positions_mask[cam] (nếu không sử dụng track) hoặc vehicle_tracker.positions_mask (nếu sử dụng track
            vehicle_masks = tracker.positions_mask
        else:
            vehicle_masks = self.detector.positions_mask[cam]
        parking_spaces_in_cam_mask = self.parking_space_initializer.positions_mask[cam]
        # Tạo hai dictionary unified_id_to_vehicle_id_ios và vehicle_id_to_unified_id_ios chứa thông tin ios (intersection over space) {unified_id1: {vehicle_id1: ..., vehicle_id2: ..., ...}, unified_id2: ...}, {vehicle_id1: {unified_id1:..., unified_id2:...,...}, vehicle_id2:...}
        unified_id_to_vehicle_id_ios = {}
        vehicle_id_to_unified_id_ios = {}

        if len(parking_spaces_in_cam) < len(vehicles_list): # Giả sử chọn vòng lặp for theo các key của dictionary của unified_id sang instance của parking_space:
            for unified_id in unified_id_to_ps:
                x_min, y_min, x_max, y_max = unified_id_to_ps[unified_id].bbox
                cropped_ps_mask = parking_spaces_in_cam_mask[y_min:y_max + 1, x_min:x_max + 1] # Crop parking space positions mask
                cropped_vh_mask = vehicle_masks[y_min:y_max + 1, x_min:x_max + 1] # Crop vehicle positions mask
                cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)
                print("unified_id", unified_id, cropped_mask.shape)
                inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=unified_id, use_unified_id=True)
                for ps_veh in inter_dict:
                    uid, vid = ps_veh
                    inter = inter_dict[ps_veh]
                    ios = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                    if ios < threshold: # Tìm số lượng giao của unified_id với các vehicle_id trong vùng crop trên và lưu vào 2 dictionary unified_id_to_vehicle_id và vehicle_id_to_unified_id nếu ios thỏa mãn > threshold đặt trước, nếu không thì bỏ qua
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
                print("vehicle_id", vehicle_id, cropped_mask.shape)
                inter_dict = find_unique_values_and_frequency(cropped_mask=cropped_mask, id=vehicle_id, use_unified_id=False)
                for ps_veh in inter_dict:
                    uid, vid = ps_veh
                    inter = inter_dict[ps_veh]
                    ios = inter / self.parking_space_initializer.square_of_mask[cam][uid]
                    if ios < threshold:
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
        print(unified_id_to_vehicle_id_ios)
        print(vehicle_id_to_unified_id_ios)
        end = time.time()
        print("This block consumes {} seconds".format(end - start))
        unified_id_status_dict = dict(zip(list(unified_id_to_ps.keys()), ["unknown"]*len(list(unified_id_to_ps.keys())))) # Tạo một unified_id_status_dict = {unified_id: "unknown", ....} tất cả các unified_id có trạng thái ban đầu là unknown
        start = time.time()
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
                            print("Parking space unified id {} and vehicle id {} is matched".format(unified_id, vehicle_id))
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
                                #adjacencies = {k: unified_id_to_row[v] for k, v in adjacencies.items()}
                                uid_match_dict["adjacencies"] = adjacencies
                                uid_match_dict["ios"] = vehicle_id_to_unified_id_ios[vehicle_id][uid_match]
                                uid_match_dict["reversed_considered_orients"] = pspace.reversed_considered_orients[cam] if cam in pspace.reversed_considered_orients else {}
                                pspace_dict[uid_match] = uid_match_dict

                            trace = []
        #                    def traverse_neighbors(uid_match):
        #                        #pspace[uid_match]["visited"] = True
        #                        if "eastern_adjacency" in pspace_dict[uid_match]["adjacencies"]:
        #                            traverse_east(uid_match)

        #                        if "western_adjacency" in pspace_dict[uid_match]["adjacencies"]:
        #                            traverse_west(uid_match)

        #                        if "southern_adjacency" in pspace_dict[uid_match]["adjacencies"]:
        #                            traverse_south(uid_match)

        #                        if "northern_adjacency" in pspace_dict[uid_match]["adjacencies"]:
        #                            traverse_north(uid_match)

        #                        if "south_east_adjacency" in pspace_dict[uid_match]["adjacencies"]:
        #                            traverse_south_east(uid_match)

        #                        if "south_west_adjacency" in pspace_dict[uid_match]["adjacencies"]:
        #                            traverse_south_west(uid_match)

        #                        if "north_west_adjacency" in pspace_dict[uid_match]["adjacencies"]:
        #                            traverse_north_west(uid_match)

        #                        if "north_east_adjacency" in pspace_dict[uid_match]["adjacencies"]:
        #                            traverse_north_east(uid_match)

        #                    def traverse_east(uid_match):
        #                        if not pspace_dict[pspace_dict[uid_match]["adjacencies"]["eastern_adjacency"]]["visited"]:
        #                            east_level = pspace_dict[uid_match]["east_level"]
        #                            south_level = pspace_dict[uid_match]["south_level"]
        #                            uid_match = pspace_dict[uid_match]["adjacencies"]["eastern_adjacency"]
        #                            pspace_dict[uid_match]["east_level"] = east_level + 1
        #                            pspace_dict[uid_match]["south_level"] = south_level
        #                            pspace_dict[uid_match]["visited"] = True
        #                            trace.append(uid_match)
        #                            traverse_neighbors(uid_match)
        #                        else:
        #                            return

        #                    def traverse_west(uid_match):
        #                        if not pspace_dict[pspace_dict[uid_match]["adjacencies"]["western_adjacency"]]["visited"]:
        #                            east_level = pspace_dict[uid_match]["east_level"]
        #                            south_level = pspace_dict[uid_match]["south_level"]
        #                            uid_match = pspace_dict[uid_match]["adjacencies"]["western_adjacency"]
        #                            pspace_dict[uid_match]["east_level"] = east_level - 1
        #                            pspace_dict[uid_match]["south_level"] = south_level
        #                            pspace_dict[uid_match]["visited"] = True
        #                            trace.append(uid_match)
        #                            traverse_neighbors(uid_match)
        #                        else:
        #                            return

        #                    def traverse_south(uid_match):
        #                        if not pspace_dict[pspace_dict[uid_match]["adjacencies"]["southern_adjacency"]]["visited"]:
        #                            east_level = pspace_dict[uid_match]["east_level"]
        #                            south_level = pspace_dict[uid_match]["south_level"]
        #                            uid_match = pspace_dict[uid_match]["adjacencies"]["southern_adjacency"]
        #                            pspace_dict[uid_match]["east_level"] = east_level
        #                            pspace_dict[uid_match]["south_level"] = south_level + 1
        #                            pspace_dict[uid_match]["visited"] = True
        #                            trace.append(uid_match)
        #                            traverse_neighbors(uid_match)
        #                        else:
        #                            return

        #                    def traverse_north(uid_match):
        #                        if not pspace_dict[pspace_dict[uid_match]["adjacencies"]["northern_adjacency"]]["visited"]:
        #                            east_level = pspace_dict[uid_match]["east_level"]
        #                            south_level = pspace_dict[uid_match]["south_level"]
        #                            uid_match = pspace_dict[uid_match]["adjacencies"]["northern_adjacency"]
        #                            pspace_dict[uid_match]["east_level"] = east_level
        #                            pspace_dict[uid_match]["south_level"] = south_level - 1
        #                            pspace_dict[uid_match]["visited"] = True
        #                            trace.append(uid_match)
        #                            traverse_neighbors(uid_match)
        #                        else:
        #                            return

        #                    def traverse_south_east(uid_match):
        #                        if not pspace_dict[pspace_dict[uid_match]["adjacencies"]["south_east_adjacency"]]["visited"]:
        #                            south_level = pspace_dict[uid_match]["south_level"]
        #                            east_level = pspace_dict[uid_match]["east_level"]
        #                            uid_match = pspace_dict[uid_match]["adjacencies"]["south_east_adjacency"]
        #                            pspace_dict[uid_match]["east_level"] = east_level + 1
        #                            pspace_dict[uid_match]["south_level"] = south_level + 1
        #                            pspace_dict[uid_match]["visited"] = True
        #                            trace.append(uid_match)
        #                            traverse_neighbors(uid_match)
        #                        else:
        #                            return

        #                    def traverse_south_west(uid_match):
        #                        if not pspace_dict[pspace_dict[uid_match]["adjacencies"]["south_west_adjacency"]]["visited"]:
        #                            south_level = pspace_dict[uid_match]["south_level"]
        #                            east_level = pspace_dict[uid_match]["east_level"]
        #                            uid_match = pspace_dict[uid_match]["adjacencies"]["south_west_adjacency"]
        #                            pspace_dict[uid_match]["east_level"] = east_level - 1
        #                            pspace_dict[uid_match]["south_level"] = south_level + 1
        #                            pspace_dict[uid_match]["visited"] = True
        #                            trace.append(uid_match)
        #                            traverse_neighbors(uid_match)
        #                        else:
        #                            return

        #                    def traverse_north_west(uid_match):
        #                        if not pspace_dict[pspace_dict[uid_match]["adjacencies"]["north_west_adjacency"]]["visited"]:
        #                            south_level = pspace_dict[uid_match]["south_level"]
        #                            east_level = pspace_dict[uid_match]["east_level"]
        #                            uid_match = pspace_dict[uid_match]["adjacencies"]["north_west_adjacency"]
        #                            pspace_dict[uid_match]["east_level"] = east_level - 1
        #                            pspace_dict[uid_match]["south_level"] = south_level - 1
        #                            pspace_dict[uid_match]["visited"] = True
        #                            trace.append(uid_match)
        #                            traverse_neighbors(uid_match)
        #                        else:
        #                            return

        #                    def traverse_north_east(uid_match):
        #                        if not pspace_dict[pspace_dict[uid_match]["adjacencies"]["north_east_adjacency"]]["visited"]:
        #                            south_level = pspace_dict[uid_match]["south_level"]
        #                            east_level = pspace_dict[uid_match]["east_level"]
        #                            uid_match = pspace_dict[uid_match]["adjacencies"]["north_east_adjacency"]
        #                            pspace_dict[uid_match]["east_level"] = east_level + 1
        #                            pspace_dict[uid_match]["south_level"] = south_level - 1
        #                            pspace_dict[uid_match]["visited"] = True
        #                            trace.append(uid_match)
        #                            traverse_neighbors(uid_match)
        #                        else:
        #                            return

        #                    random_row = np.random.choice(list(cols_to_rows[col].keys()))

        #                    pspace_dict[random_row]["visited"] = True # Choose random_row as starting point
        #                    trace.append(random_row) # Initialize track
        #                    traverse_neighbors(random_row) # Assign value to east_level and south_level

        #                    #print("Random row {}, trace {}".format(random_row, trace))

        #                    pspace_dict = dict(sorted(pspace_dict.items(), key=lambda s: s[1]["east_level"]))
        #                    reversed_considered_orients = {}
        #                    for uid_match in pspace_dict:
        #                        orients = pspace_dict[uid_match]["reversed_considered_orients"]
        #                        for orient in orients:
        #                            #if convert_considered_orient_dict[orient] in pspace_dict[uid_match]["adjacencies"]:
        #                            if orient not in reversed_considered_orients:
        #                                reversed_considered_orients[orient] = []
        #                            reversed_considered_orients[orient].append(uid_match)

        #                    #print("Reversed_considered_orients {}".format(reversed_considered_orients))

        #                    considered_east_west_row_list = []
        #                    max_east = False
        #                    if "north_east" in reversed_considered_orients or "east" in reversed_considered_orients \
        #                          or "south_east" in reversed_considered_orients:
        #                        min_east_level = pspace_dict[min(pspace_dict.keys(), key=lambda x: pspace_dict[x]["east_level"])]["east_level"]
        #                        considered_rows = list(dict(filter(lambda x: x[1]["east_level"] == min_east_level, pspace_dict.items())).keys())
        #                        considered_east_west_row_list.extend(considered_rows)
        #                        #print("min_east_level {}, consider_rows {}, considered_east_west_row_list {}, max_east {}".format(min_east_level, considered_rows, considered_east_west_row_list, max_east))

        #                    elif "north_west"  in reversed_considered_orients or "west" in reversed_considered_orients \
        #                            or "south_west" in reversed_considered_orients:
        #                        max_east_level = pspace_dict[max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["east_level"])]["east_level"]
        #                        considered_rows = list(dict(filter(lambda x: x[1]["east_level"] == max_east_level, pspace_dict.items())).keys())
        #                        considered_east_west_row_list.extend(considered_rows)
        #                        max_east = True
        #                        #print("max_east_level {}, consider_rows {}, considered_east_west_row_list {}, max_east {}".format(max_east_level, considered_rows, considered_east_west_row_list, max_east))

        #                    considered_south_north_row_list = []
        #                    max_south = True
        #                    if "north" in reversed_considered_orients or "north_west" in reversed_considered_orients \
        #                            or "north_east" in reversed_considered_orients:
        #                        max_south_level = pspace_dict[max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["south_level"])]["south_level"]
        #                        considered_rows = list(dict(filter(lambda x: x[1]["south_level"] == max_south_level, pspace_dict.items())).keys())
        #                        considered_south_north_row_list.extend(considered_rows)
        #                        #print("max_south_level {}, considered_rows {}, considered_south_north_row_list{}, max_south {}".format(max_south_level, considered_rows, considered_south_north_row_list, max_south))

        #                    elif "south" in reversed_considered_orients or "south_west" in reversed_considered_orients \
        #                            or "south_east" in reversed_considered_orients:
        #                        min_south_level = pspace_dict[min(pspace_dict.keys(), key=lambda x: pspace_dict[x]["south_level"])]["south_level"]
        #                        considered_rows = list(dict(filter(lambda x: x[1]["south_level"] == min_south_level, pspace_dict.items())).keys())
        #                        considered_south_north_row_list.extend(considered_rows)
        #                        max_south = False
        #                        #print("min_south_level {}, considered_rows {}, considered_south_north_row_list{}, max_south {}".format(min_south_level, considered_rows, considered_south_north_row_list, max_south))

        #                    considered_row = list(set(considered_east_west_row_list).intersection(considered_south_north_row_list))

        #                    if len(considered_row) == 0:
        #                        if len(considered_east_west_row_list) > 0 or len(considered_south_north_row_list) > 0: # Parking space does not belong to any reversed considered orients
        #                            chosen_row = max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["ios"])
        #                            rows_status_dict[chosen_row] = "filled"
        #                            for uid_match in pspace_dict:
        #                                if uid_match != chosen_row:
        #                                    if pspace_dict[uid_match]["ios"] > 0.75:
        #                                        rows_status_dict[uid_match] = "unknown"
        #                                    else:
        #                                        rows_status_dict[uid_match] = "available"
        #                        else:
        #                            for uid_match in pspace_dict:
        #                                if pspace_dict[uid_match]["ios"] > 0.65:
        #                                    rows_status_dict[uid_match] = "filled"
        #                    else:
        #                        assert len(considered_row) == 1
        #                        considered_row = considered_row[0]
        #                        filled_list = []
        #                        rows_status_dict[considered_row] = "filled"
        #                        filled_list.append(considered_row)
        #                        if max_south:
        #                            if "northern_adjacency" in pspace_dict[considered_row]["adjacencies"]:
        #                                north_of_considered_row = pspace_dict[considered_row]["adjacencies"]["northern_adjacency"]
        #                                if pspace_dict[north_of_considered_row]["ios"] > 0.6: # and vehicles_list[col].class_id == 1 # "truck"
        #                                    rows_status_dict[north_of_considered_row] = "filled"
        #                                    filled_list.append(north_of_considered_row)
        #                        else:
        #                            if "southern_adjacency" in pspace_dict[considered_row]["adjacencies"]:
        #                                south_of_considered_row = pspace_dict[considered_row]["adjacencies"]["southern_adjacency"]
        #                                if pspace_dict[south_of_considered_row]["ios"] > 0.6: # and vehicles_list[col].class_id == 1 # "truck"
        #                                    rows_status_dict[south_of_considered_row] = "filled"
        #                                    filled_list.append(south_of_considered_row)
        #                        for uid_match in pspace_dict:
        #                            if uid_match not in filled_list:
        #                                if pspace_dict[uid_match]["ios"] > 0.7:
        #                                    rows_status_dict[uid_match] = "unknown"
        #                                else:
        #                                    rows_status_dict[uid_match] = "available"

        #                    #print("Row {}, Col {}, Pspace_dict {}".format(row, col, pspace_dict))

        #                considered_col_list.append(col)
        #unified_id_status_dict = {row_to_unified_id[k]:v for k, v in rows_status_dict.items()}
        end = time.time()
        print("This block consumes {} seconds".format(end - start))
        #start = time.time()
        #color_mask = np.zeros_like(frame, dtype=np.uint8)
        #for row in rows_status_dict:
        #   if rows_status_dict[row] == "filled":
        #       color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 0, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #   elif rows_status_dict[row] == "unknown":
        #       color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 255, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #   else:
        #       color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 255, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #for vehicle_mask in vehicle_masks:
        #   color_mask = np.where(vehicle_mask[:, :, np.newaxis], np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)

        #frame = np.where(color_mask > 0, cv2.addWeighted(frame, 0.4, color_mask, 0.6, 0), frame)
        #end = time.time()
        #print("This block consumes {} seconds".format(end - start))
        #return vehicles_list, parking_spaces_in_cam, ios, frame

    def image_match(self, image_path, save_dir, cam="cam_1", threshold=0.3, is_tracking=False, is_showimage=True):
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
        vehicles, parking_spaces, ios, frame = self.frame_match(frame=image, cam=cam, threshold=threshold, is_tracking=is_tracking, tracker=tracker)
        if is_showimage:
            cv2.imshow("", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        results_path = os.path.join(save_dir, os.path.basename(image_path).split(".")[0] + ".jpg")
        cv2.imwrite(results_path, frame)

    def video_match(self, video_source, is_savevideo=False, save_dir=None, cam="cam_1", threshold=0.3, is_tracking=True, is_showframe=True):
        if is_tracking:
            tracker = VehicleTracker(detection_vehicle_thresh=0.2,
                                     inactive_steps_before_removed=10,
                                     reid_iou_threshold=0.3,
                                     max_traject_steps=50,
                                     parking_ground=self.parking_ground,
                                     cam=cam)
        else:
            tracker = None

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

                vehicles, parking_spaces, ios, frame = self.frame_match(frame=frame,
                                                                        cam=cam,
                                                                        threshold=threshold,
                                                                        is_tracking=is_tracking,
                                                                        tracker=tracker)
                if is_savevideo:
                    output.write(frame)
                if is_showframe:
                    cv2.imshow("", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stopped = True
            else:
                if is_savevideo:
                    output.release()
                    print("Save video and exit")
                else:
                    print("Exit")
                cv2.destroyAllWindows()
                break


matcher = Matcher()
tracker = VehicleTracker(detection_vehicle_thresh=0.2,
                         inactive_steps_before_removed=10,
                         reid_iou_threshold=0.3,
                         max_traject_steps=50,
                         parking_ground="parking_ground_SA",
                         cam="cam_2")
image_path = "../test_object_detection_models/images/201909_20190914_2_2019-09-14_05-00-00_8987.jpg"
demo_images_dir = r"F:\Parking_Spaces_Recommendation_Data\demo_images"
image = cv2.imread(image_path)
matcher.frame_match(frame=image,
                    cam="cam_2",
                    threshold=0.3,
                    is_tracking=True,
                    tracker=tracker)
