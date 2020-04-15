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
                 parking_ground="parking_ground_SA",
                 active_cams=["cam_1"],
                 shape=(720, 1280),
                 config_json_path=os.path.join(ROOT_DIR, "parking_spaces_data/parking_spaces_unified_id_segmen_in_cameras.json")):
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
        self.detector = VehicleDetector(checkpoint_name=checkpoint_name)
        self.parking_spaces_list = self.parking_space_initializer.initialize_parking_spaces()

    def frame_match(self, frame, cam="cam_1"):

        assert cam in self.active_cams

        detections_list = self.detector(frame=frame, parking_ground=self.parking_ground, cam=cam)

        parking_spaces_in_cam = list(filter(lambda x: cam in list(x.positions.keys()), self.parking_spaces_list))

        col_to_det_id = dict(zip(list(range(len(detections_list))), list(map(lambda x: x.detection_id, detections_list))))

        row_to_unified_id = dict(zip(list(range(len(parking_spaces_in_cam))), list(map(lambda x: x.unified_id, parking_spaces_in_cam))))
        unified_id_to_row = {v: k for k, v in row_to_unified_id.items()}
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
        num_cols = ios.shape[1]
        cols_to_rows = {}
        for i in range(num_cols):
            cols_i = ios[:, i]
            sorted_ind = np.argsort(cols_i)[::-1]
            sorted_ind_above_thresh = sorted_ind[cols_i[sorted_ind] > 0.2]
            if len(sorted_ind_above_thresh.tolist()) > 0:
                cols_to_rows[i] = dict(zip(sorted_ind_above_thresh.tolist(), cols_i[sorted_ind_above_thresh].tolist()))
            print("column (vehicle) {} sorted_ind_above_thresh: {}".format(i, dict(zip(list(map(lambda x: row_to_unified_id[x], sorted_ind_above_thresh)), cols_i[sorted_ind_above_thresh].tolist()))))
        print("Dictionary cols to rows: {}".format(cols_to_rows))
        num_rows = ios.shape[0]
        rows_to_cols = {}
        for j in range(num_rows):
            rows_j = ios[j, :]
            sorted_ind = np.argsort(rows_j)[::-1]
            sorted_ind_above_thresh = sorted_ind[rows_j[sorted_ind] > 0.2]
            if len(sorted_ind_above_thresh.tolist()) > 0:
                rows_to_cols[j] = dict(zip(sorted_ind_above_thresh.tolist(), rows_j[sorted_ind_above_thresh].tolist()))
            print("row (parking space) {}-{} sorted_ind_above_thresh: {}".format(j, row_to_unified_id[j], dict(zip(sorted_ind_above_thresh.tolist(), rows_j[sorted_ind_above_thresh].tolist()))))
        print("Dictionary rows to cols: {}".format(rows_to_cols))

        rows_status_dict = dict(zip(list(row_to_unified_id.keys()), ["unknown"]*len(list(row_to_unified_id.keys()))))
        #for row in rows_status_dict:
        #    if row in rows_to_cols:
        #        rows_status_dict[row] = "filled"
        #    else:
        #        rows_status_dict[row] = "available"
        #unified_id_status_dict = {row_to_unified_id[k]:v for k, v in rows_status_dict.items()}
#
        #color_mask = np.zeros_like(frame, dtype=np.uint8)
        #for row in rows_status_dict:
        #    if rows_status_dict[row] == "filled":
        #        color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 0, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #    elif rows_status_dict[row] == "unknown":
        #        color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 255, 255], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #    else:
        #        color_mask = np.where(parking_spaces_in_cam_mask[row][:, :, np.newaxis], np.array([0, 255, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        #for detection_mask in detection_masks:
        #    color_mask = np.where(detection_mask[:, :, np.newaxis], np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
#
        #frame = np.where(color_mask > 0, cv2.addWeighted(frame, 0.4, color_mask, 0.6, 0), frame)
#
        #cv2.imshow("", frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
#
        #k = 0
        #demo_images_dir = r"F:\Parking_Spaces_Recommendation_Data\demo_images"
        #for dirs, _, files in os.walk(demo_images_dir):
        #    for file in files:
        #        if file.endswith((".jpg", ".png")):
        #            k += 1
        #cv2.imwrite(os.path.join(demo_images_dir, str(k) + ".jpg"), frame)

        considered_col_list = []
        for row in rows_status_dict:
            if row not in rows_to_cols:
                rows_status_dict[row] = "available"
            else:
                for col in rows_to_cols[row]:
                    if col not in considered_col_list:
                        if len(cols_to_rows[col]) == 1:
                            assert row in cols_to_rows[col]
                            rows_status_dict[row] = "available"
                            print("Row {} and col {} is matched".format(row, col))
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
                            trace.append(random_row)
                            traverse_neighbors(random_row)

                            print("Random row {}, trace {}".format(random_row, trace))

                            print("Row {}, Col {}, Pspace_dict {}".format(row, col, pspace_dict))

                        considered_col_list.append(col)
        return detections_list, parking_spaces_in_cam, ios, iov, iou

matcher = Matcher()
image = cv2.imread(os.path.join(ROOT_DIR, "test_object_detection_models/images/201909_20190914_1_2019-09-14_01-00-00_8990.jpg"))
vehicles, parking_spaces, ios, iov, iou = matcher.frame_match(image)
