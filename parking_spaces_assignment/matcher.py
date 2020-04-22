import sys
import os
from tqdm import tqdm
import numpy as np
import cv2
from vehicle_tracking.vehicle_detector import VehicleDetector
from parking_spaces_assignment.parking_space import ParkingSpacesInitializer
from vehicle_tracking.vehicle_tracker import VehicleTracker
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

        vehicles_list = self.detector(frame=frame, parking_ground=self.parking_ground, cam=cam)

        if is_tracking:
            assert tracker, "vehicles tracker cannot be None"
            tracker.step(vehicle_detections=vehicles_list)
            vehicles_list = tracker.get_result()
        start = time.time()
        parking_spaces_in_cam = list(filter(lambda x: cam in list(x.positions.keys()), self.parking_spaces_list))
        if not is_tracking:
            col_to_veh_id = dict(zip(list(range(len(vehicles_list))), list(map(lambda x: x.detection_id, vehicles_list))))
        else:
            col_to_veh_id = dict(zip(list(range(len(vehicles_list))), list(map(lambda x: x.track_id, vehicles_list))))

        row_to_unified_id = dict(zip(list(range(len(parking_spaces_in_cam))), list(map(lambda x: x.unified_id, parking_spaces_in_cam))))
        unified_id_to_row = {v: k for k, v in row_to_unified_id.items()}
        #print(col_to_veh_id, row_to_unified_id)

        vehicle_masks = np.stack(list(map(lambda x: x.mask, vehicles_list)), axis=0)
        parking_spaces_in_cam_mask = np.stack(list(map(lambda x: x.positions_mask[cam], parking_spaces_in_cam)), axis=0)

        intersection = np.logical_and(parking_spaces_in_cam_mask[:, np.newaxis, :, :], vehicle_masks[np.newaxis, :, :, :])
        intersection = np.count_nonzero(intersection, axis=(2, 3)).astype(np.float32)
        ios = intersection / np.count_nonzero(parking_spaces_in_cam_mask, axis=(1, 2))[:, np.newaxis]
        num_cols = ios.shape[1]
        cols_to_rows = {}
        for i in range(num_cols):
            cols_i = ios[:, i]
            sorted_ind = np.argsort(cols_i)[::-1]
            sorted_ind_above_thresh = sorted_ind[cols_i[sorted_ind] > threshold]
            if len(sorted_ind_above_thresh.tolist()) > 0:
                cols_to_rows[i] = dict(zip(sorted_ind_above_thresh.tolist(), cols_i[sorted_ind_above_thresh].tolist()))
            #print("column (vehicle) {} sorted_ind_above_thresh: {}".format(i, dict(zip(list(map(lambda x: row_to_unified_id[x], sorted_ind_above_thresh)), cols_i[sorted_ind_above_thresh].tolist()))))
        #print("Dictionary cols to rows: {}".format(cols_to_rows))
        num_rows = ios.shape[0]
        rows_to_cols = {}
        for j in range(num_rows):
            rows_j = ios[j, :]
            sorted_ind = np.argsort(rows_j)[::-1]
            sorted_ind_above_thresh = sorted_ind[rows_j[sorted_ind] > threshold]
            if len(sorted_ind_above_thresh.tolist()) > 0:
                rows_to_cols[j] = dict(zip(sorted_ind_above_thresh.tolist(), rows_j[sorted_ind_above_thresh].tolist()))
            #print("row (parking space) {}-{} sorted_ind_above_thresh: {}".format(j, row_to_unified_id[j], dict(zip(sorted_ind_above_thresh.tolist(), rows_j[sorted_ind_above_thresh].tolist()))))
        #print("Dictionary rows to cols: {}".format(rows_to_cols))
        end = time.time()
        print("This block consumes {} seconds".format(end - start))
        rows_status_dict = dict(zip(list(row_to_unified_id.keys()), ["unknown"]*len(list(row_to_unified_id.keys()))))
        start = time.time()
        considered_col_list = []
        for row in rows_status_dict:
            if row not in rows_to_cols:
                rows_status_dict[row] = "available"
            else:
                for col in rows_to_cols[row]:
                    if col not in considered_col_list:
                        if len(cols_to_rows[col]) == 1:
                            assert row in cols_to_rows[col]
                            rows_status_dict[row] = "filled"
                            #print("Row {} and col {} is matched".format(row, col))
                        else:
                            assert row in cols_to_rows[col]
                            pspace_dict = {}  # Dictionary of dictionary, each dictionary represents and parking space with information east level, south level and adjacencies against orient
                            for row_match in cols_to_rows[col]:
                                row_match_dict = {}
                                row_match_dict["east_level"] = 0
                                row_match_dict["south_level"] = 0
                                row_match_dict["visited"] = False
                                pspace = parking_spaces_in_cam[row_match]
                                adjacencies = dict(filter(lambda x: x[1] is not None, pspace.adjacencies.items()))
                                adjacencies = dict(filter(lambda x: unified_id_to_row[x[1]] in cols_to_rows[col], adjacencies.items()))
                                adjacencies = {k: unified_id_to_row[v] for k, v in adjacencies.items()}
                                row_match_dict["adjacencies"] = adjacencies
                                row_match_dict["ios"] = cols_to_rows[col][row_match]
                                row_match_dict["reversed_considered_orients"] = pspace.reversed_considered_orients[cam] if cam in pspace.reversed_considered_orients else {}
                                pspace_dict[row_match] = row_match_dict

                            trace = []
                            def traverse_neighbors(row_match):
                                #pspace[row_match]["visited"] = True
                                if "eastern_adjacency" in pspace_dict[row_match]["adjacencies"]:
                                    traverse_east(row_match)

                                if "western_adjacency" in pspace_dict[row_match]["adjacencies"]:
                                    traverse_west(row_match)

                                if "southern_adjacency" in pspace_dict[row_match]["adjacencies"]:
                                    traverse_south(row_match)

                                if "northern_adjacency" in pspace_dict[row_match]["adjacencies"]:
                                    traverse_north(row_match)

                                if "south_east_adjacency" in pspace_dict[row_match]["adjacencies"]:
                                    traverse_south_east(row_match)

                                if "south_west_adjacency" in pspace_dict[row_match]["adjacencies"]:
                                    traverse_south_west(row_match)

                                if "north_west_adjacency" in pspace_dict[row_match]["adjacencies"]:
                                    traverse_north_west(row_match)

                                if "north_east_adjacency" in pspace_dict[row_match]["adjacencies"]:
                                    traverse_north_east(row_match)

                            def traverse_east(row_match):
                                if not pspace_dict[pspace_dict[row_match]["adjacencies"]["eastern_adjacency"]]["visited"]:
                                    east_level = pspace_dict[row_match]["east_level"]
                                    south_level = pspace_dict[row_match]["south_level"]
                                    row_match = pspace_dict[row_match]["adjacencies"]["eastern_adjacency"]
                                    pspace_dict[row_match]["east_level"] = east_level + 1
                                    pspace_dict[row_match]["south_level"] = south_level
                                    pspace_dict[row_match]["visited"] = True
                                    trace.append(row_match)
                                    traverse_neighbors(row_match)
                                else:
                                    return

                            def traverse_west(row_match):
                                if not pspace_dict[pspace_dict[row_match]["adjacencies"]["western_adjacency"]]["visited"]:
                                    east_level = pspace_dict[row_match]["east_level"]
                                    south_level = pspace_dict[row_match]["south_level"]
                                    row_match = pspace_dict[row_match]["adjacencies"]["western_adjacency"]
                                    pspace_dict[row_match]["east_level"] = east_level - 1
                                    pspace_dict[row_match]["south_level"] = south_level
                                    pspace_dict[row_match]["visited"] = True
                                    trace.append(row_match)
                                    traverse_neighbors(row_match)
                                else:
                                    return

                            def traverse_south(row_match):
                                if not pspace_dict[pspace_dict[row_match]["adjacencies"]["southern_adjacency"]]["visited"]:
                                    east_level = pspace_dict[row_match]["east_level"]
                                    south_level = pspace_dict[row_match]["south_level"]
                                    row_match = pspace_dict[row_match]["adjacencies"]["southern_adjacency"]
                                    pspace_dict[row_match]["east_level"] = east_level
                                    pspace_dict[row_match]["south_level"] = south_level + 1
                                    pspace_dict[row_match]["visited"] = True
                                    trace.append(row_match)
                                    traverse_neighbors(row_match)
                                else:
                                    return

                            def traverse_north(row_match):
                                if not pspace_dict[pspace_dict[row_match]["adjacencies"]["northern_adjacency"]]["visited"]:
                                    east_level = pspace_dict[row_match]["east_level"]
                                    south_level = pspace_dict[row_match]["south_level"]
                                    row_match = pspace_dict[row_match]["adjacencies"]["northern_adjacency"]
                                    pspace_dict[row_match]["east_level"] = east_level
                                    pspace_dict[row_match]["south_level"] = south_level - 1
                                    pspace_dict[row_match]["visited"] = True
                                    trace.append(row_match)
                                    traverse_neighbors(row_match)
                                else:
                                    return

                            def traverse_south_east(row_match):
                                if not pspace_dict[pspace_dict[row_match]["adjacencies"]["south_east_adjacency"]]["visited"]:
                                    south_level = pspace_dict[row_match]["south_level"]
                                    east_level = pspace_dict[row_match]["east_level"]
                                    row_match = pspace_dict[row_match]["adjacencies"]["south_east_adjacency"]
                                    pspace_dict[row_match]["east_level"] = east_level + 1
                                    pspace_dict[row_match]["south_level"] = south_level + 1
                                    pspace_dict[row_match]["visited"] = True
                                    trace.append(row_match)
                                    traverse_neighbors(row_match)
                                else:
                                    return

                            def traverse_south_west(row_match):
                                if not pspace_dict[pspace_dict[row_match]["adjacencies"]["south_west_adjacency"]]["visited"]:
                                    south_level = pspace_dict[row_match]["south_level"]
                                    east_level = pspace_dict[row_match]["east_level"]
                                    row_match = pspace_dict[row_match]["adjacencies"]["south_west_adjacency"]
                                    pspace_dict[row_match]["east_level"] = east_level - 1
                                    pspace_dict[row_match]["south_level"] = south_level + 1
                                    pspace_dict[row_match]["visited"] = True
                                    trace.append(row_match)
                                    traverse_neighbors(row_match)
                                else:
                                    return

                            def traverse_north_west(row_match):
                                if not pspace_dict[pspace_dict[row_match]["adjacencies"]["north_west_adjacency"]]["visited"]:
                                    south_level = pspace_dict[row_match]["south_level"]
                                    east_level = pspace_dict[row_match]["east_level"]
                                    row_match = pspace_dict[row_match]["adjacencies"]["north_west_adjacency"]
                                    pspace_dict[row_match]["east_level"] = east_level - 1
                                    pspace_dict[row_match]["south_level"] = south_level - 1
                                    pspace_dict[row_match]["visited"] = True
                                    trace.append(row_match)
                                    traverse_neighbors(row_match)
                                else:
                                    return

                            def traverse_north_east(row_match):
                                if not pspace_dict[pspace_dict[row_match]["adjacencies"]["north_east_adjacency"]]["visited"]:
                                    south_level = pspace_dict[row_match]["south_level"]
                                    east_level = pspace_dict[row_match]["east_level"]
                                    row_match = pspace_dict[row_match]["adjacencies"]["north_east_adjacency"]
                                    pspace_dict[row_match]["east_level"] = east_level + 1
                                    pspace_dict[row_match]["south_level"] = south_level - 1
                                    pspace_dict[row_match]["visited"] = True
                                    trace.append(row_match)
                                    traverse_neighbors(row_match)
                                else:
                                    return

                            random_row = np.random.choice(list(cols_to_rows[col].keys()))

                            pspace_dict[random_row]["visited"] = True # Choose random_row as starting point
                            trace.append(random_row) # Initialize track
                            traverse_neighbors(random_row) # Assign value to east_level and south_level

                            #print("Random row {}, trace {}".format(random_row, trace))

                            pspace_dict = dict(sorted(pspace_dict.items(), key=lambda s: s[1]["east_level"]))
                            reversed_considered_orients = {}
                            for row_match in pspace_dict:
                                orients = pspace_dict[row_match]["reversed_considered_orients"]
                                for orient in orients:
                                    #if convert_considered_orient_dict[orient] in pspace_dict[row_match]["adjacencies"]:
                                    if orient not in reversed_considered_orients:
                                        reversed_considered_orients[orient] = []
                                    reversed_considered_orients[orient].append(row_match)

                            #print("Reversed_considered_orients {}".format(reversed_considered_orients))

                            considered_east_west_row_list = []
                            max_east = False
                            if "north_east" in reversed_considered_orients or "east" in reversed_considered_orients \
                                  or "south_east" in reversed_considered_orients:
                                min_east_level = pspace_dict[min(pspace_dict.keys(), key=lambda x: pspace_dict[x]["east_level"])]["east_level"]
                                considered_rows = list(dict(filter(lambda x: x[1]["east_level"] == min_east_level, pspace_dict.items())).keys())
                                considered_east_west_row_list.extend(considered_rows)
                                #print("min_east_level {}, consider_rows {}, considered_east_west_row_list {}, max_east {}".format(min_east_level, considered_rows, considered_east_west_row_list, max_east))

                            elif "north_west"  in reversed_considered_orients or "west" in reversed_considered_orients \
                                    or "south_west" in reversed_considered_orients:
                                max_east_level = pspace_dict[max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["east_level"])]["east_level"]
                                considered_rows = list(dict(filter(lambda x: x[1]["east_level"] == max_east_level, pspace_dict.items())).keys())
                                considered_east_west_row_list.extend(considered_rows)
                                max_east = True
                                #print("max_east_level {}, consider_rows {}, considered_east_west_row_list {}, max_east {}".format(max_east_level, considered_rows, considered_east_west_row_list, max_east))

                            considered_south_north_row_list = []
                            max_south = True
                            if "north" in reversed_considered_orients or "north_west" in reversed_considered_orients \
                                    or "north_east" in reversed_considered_orients:
                                max_south_level = pspace_dict[max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["south_level"])]["south_level"]
                                considered_rows = list(dict(filter(lambda x: x[1]["south_level"] == max_south_level, pspace_dict.items())).keys())
                                considered_south_north_row_list.extend(considered_rows)
                                #print("max_south_level {}, considered_rows {}, considered_south_north_row_list{}, max_south {}".format(max_south_level, considered_rows, considered_south_north_row_list, max_south))

                            elif "south" in reversed_considered_orients or "south_west" in reversed_considered_orients \
                                    or "south_east" in reversed_considered_orients:
                                min_south_level = pspace_dict[min(pspace_dict.keys(), key=lambda x: pspace_dict[x]["south_level"])]["south_level"]
                                considered_rows = list(dict(filter(lambda x: x[1]["south_level"] == min_south_level, pspace_dict.items())).keys())
                                considered_south_north_row_list.extend(considered_rows)
                                max_south = False
                                #print("min_south_level {}, considered_rows {}, considered_south_north_row_list{}, max_south {}".format(min_south_level, considered_rows, considered_south_north_row_list, max_south))

                            considered_row = list(set(considered_east_west_row_list).intersection(considered_south_north_row_list))

                            if len(considered_row) == 0:
                                if len(considered_east_west_row_list) > 0 or len(considered_south_north_row_list) > 0: # Parking space does not belong to any reversed considered orients
                                    chosen_row = max(pspace_dict.keys(), key=lambda x: pspace_dict[x]["ios"])
                                    rows_status_dict[chosen_row] = "filled"
                                    for row_match in pspace_dict:
                                        if row_match != chosen_row:
                                            if pspace_dict[row_match]["ios"] > 0.75:
                                                rows_status_dict[row_match] = "unknown"
                                            else:
                                                rows_status_dict[row_match] = "available"
                                else:
                                    for row_match in pspace_dict:
                                        if pspace_dict[row_match]["ios"] > 0.65:
                                            rows_status_dict[row_match] = "filled"
                            else:
                                assert len(considered_row) == 1
                                considered_row = considered_row[0]
                                filled_list = []
                                rows_status_dict[considered_row] = "filled"
                                filled_list.append(considered_row)
                                if max_south:
                                    if "northern_adjacency" in pspace_dict[considered_row]["adjacencies"]:
                                        north_of_considered_row = pspace_dict[considered_row]["adjacencies"]["northern_adjacency"]
                                        if pspace_dict[north_of_considered_row]["ios"] > 0.6: # and vehicles_list[col].class_id == 1 # "truck"
                                            rows_status_dict[north_of_considered_row] = "filled"
                                            filled_list.append(north_of_considered_row)
                                else:
                                    if "southern_adjacency" in pspace_dict[considered_row]["adjacencies"]:
                                        south_of_considered_row = pspace_dict[considered_row]["adjacencies"]["southern_adjacency"]
                                        if pspace_dict[south_of_considered_row]["ios"] > 0.6: # and vehicles_list[col].class_id == 1 # "truck"
                                            rows_status_dict[south_of_considered_row] = "filled"
                                            filled_list.append(south_of_considered_row)
                                for row_match in pspace_dict:
                                    if row_match not in filled_list:
                                        if pspace_dict[row_match]["ios"] > 0.7:
                                            rows_status_dict[row_match] = "unknown"
                                        else:
                                            rows_status_dict[row_match] = "available"

                            #print("Row {}, Col {}, Pspace_dict {}".format(row, col, pspace_dict))

                        considered_col_list.append(col)
        unified_id_status_dict = {row_to_unified_id[k]:v for k, v in rows_status_dict.items()}
        end = time.time()
        print("This block consumes {} seconds".format(end - start))
        start = time.time()
        color_mask = np.zeros_like(frame, dtype=np.uint8)
        for row in rows_status_dict:
           if rows_status_dict[row] == "filled":
               color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 0, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
           elif rows_status_dict[row] == "unknown":
               color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 255, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
           else:
               color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 255, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        for vehicle_mask in vehicle_masks:
           color_mask = np.where(vehicle_mask[:, :, np.newaxis], np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)

        frame = np.where(color_mask > 0, cv2.addWeighted(frame, 0.4, color_mask, 0.6, 0), frame)
        end = time.time()
        print("This block consumes {} seconds".format(end - start))
        return vehicles_list, parking_spaces_in_cam, ios, frame

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


#matcher = Matcher()
#image_path = "../test_object_detection_models/images/201909_20190914_2_2019-09-14_05-00-00_8987.jpg"
#demo_images_dir = r"F:\Parking_Spaces_Recommendation_Data\demo_images"
#
#matcher.image_match(image_path=image_path,
#                    save_dir=demo_images_dir,
#                    cam="cam_2",
#                    threshold=0.3,
#                    is_tracking=True)
