import sys
import os
import argparse
import codecs, json
import numpy as np
from skimage.draw import polygon

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

from parking_spaces_assignment.parking_space import ParkingSpace, ParkingSpacesInitializer


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="deploy_serving/jsons", help="Directory contains position mask and square mask of parking spaces")
    parser.add_argument("--parking_grounds_list", type=str, default="parking_ground_SA,parking_ground_PA")

    args = parser.parse_args()

    return args


class JsonParkingSpacesInitializer(ParkingSpacesInitializer):
    def __init__(self, active_cams: list, parking_ground="parking_ground_SA", shape=(720, 1280), config_json_path="../parking_spaces_data/parking_spaces_unified_id_segmen_in_cameras.json"):
        super(JsonParkingSpacesInitializer, self).__init__(active_cams=active_cams, parking_ground=parking_ground, shape=shape, config_json_path=config_json_path)

    def initialize_parking_spaces(self, jsons_mask):
        parking_spaces_list = []
        outlier_parking_spaces_list = []
        for unified_id in self.unified_id_list:
            if unified_id < 1000:
                positions = self.config_json[self.parking_ground][str(unified_id)][
                    "positions"]  # Sau khi sửa json sẽ phải thêm ["parking_ground_SA"] vào sau config_json
                reversed_considered_orients = self.config_json[self.parking_ground][str(unified_id)][
                    "reversed_considered_orients"]
                adjacencies = self.config_json[self.parking_ground][str(unified_id)]["adjacencies"]
                type_space = self.config_json[self.parking_ground][str(unified_id)]["type_space"]
                considered_in_cam = self.config_json[self.parking_ground][str(unified_id)]["considered_in_cam"]
                parking_spaces_list.append(ParkingSpace(unified_id=unified_id,
                                                        positions=positions,
                                                        reversed_considered_orients=reversed_considered_orients,
                                                        adjacencies=adjacencies,
                                                        type_space=type_space,
                                                        considered_in_cam=considered_in_cam,
                                                        active_cams=self.active_cams,
                                                        shape=self.shape))
                jsons_mask[self.parking_ground][unified_id] = {}
                jsons_mask[self.parking_ground][unified_id]["positions"] = positions
                jsons_mask[self.parking_ground][unified_id]["reversed_considered_orients"] = reversed_considered_orients
                jsons_mask[self.parking_ground][unified_id]["adjacencies"] = adjacencies
                jsons_mask[self.parking_ground][unified_id]["type_space"] = type_space
                jsons_mask[self.parking_ground][unified_id]["considered_in_cam"] = considered_in_cam
                for cam in positions:
                    if cam in self.active_cams and cam == considered_in_cam:
                        cc, rr = np.array(positions[cam], dtype=np.uint16).reshape(-1, 2).T
                        rr, cc = polygon(rr, cc)
                        self.positions_mask[cam][rr, cc] = unified_id
                        self.square_of_mask[cam][unified_id] = rr.shape[
                            0]  # Square of mask (number of pixel) of parking space unified id in camera
            else:
                positions = self.config_json[self.parking_ground][str(unified_id)][
                    "positions"]  # Sau khi sửa json sẽ phải thêm ["parking_ground_SA"] vào sau config_json
                type_space = self.config_json[self.parking_ground][str(unified_id)]["type_space"]
                considered_in_cam = self.config_json[self.parking_ground][str(unified_id)]["considered_in_cam"]
                outlier_parking_spaces_list.append(ParkingSpace(unified_id=unified_id,
                                                                positions=positions,
                                                                reversed_considered_orients=None,
                                                                adjacencies=None,
                                                                type_space=type_space,
                                                                considered_in_cam=considered_in_cam,
                                                                active_cams=self.active_cams,
                                                                shape=self.shape))

                for cam in positions:
                    if cam in self.active_cams and cam == considered_in_cam:
                        cc, rr = np.array(positions[cam], dtype=np.uint16).reshape(-1, 2).T
                        rr, cc = polygon(rr, cc)
                        self.outlier_positions_mask[cam][rr, cc] = unified_id
                        self.outlier_square_of_mask[cam][unified_id] = rr.shape[
                            0]  # Square of mask (number of pixel) of outlier parking space unified id in camera

        return sorted(parking_spaces_list, key=lambda x: x.unified_id), sorted(outlier_parking_spaces_list,
                                                                               key=lambda x: x.unified_id)


if __name__ == '__main__':

    args = get_args()

    if not os.path.exists(os.path.join(ROOT_DIR, args.save_dir)):
        os.makedirs(os.path.join(ROOT_DIR, args.save_dir), exist_ok=True)

    parking_grounds_list = args.parking_grounds_list.split(",")

    jsons_mask = {}

    shape = (720, 1280)
    config_json_path = os.path.join(ROOT_DIR, "parking_spaces_data/parking_spaces_unified_id_segmen_in_cameras.json")

    for parking_ground in parking_grounds_list:
        jsons_mask[parking_ground] = {}

        if parking_ground == "parking_ground_SA":
            active_cams = ["cam_1", "cam_2", "cam_3"]
        else:
            active_cams = ["cam_1", "cam_2", "cam_3", "cam_4"]

        parking_space_initializer = JsonParkingSpacesInitializer(active_cams=active_cams,
                                                                 parking_ground=parking_ground,
                                                                 shape=shape,
                                                                 config_json_path=config_json_path)

        parking_spaces_list, outlier_parking_spaces_list = parking_space_initializer.initialize_parking_spaces(jsons_mask=jsons_mask)

        jsons_mask[parking_ground]["positions_mask"] = {}
        jsons_mask[parking_ground]["square_of_mask"] = {}

        jsons_mask[parking_ground]["outlier_positions_mask"] = {}
        jsons_mask[parking_ground]["outlier_square_of_mask"] = {}

        for cam in active_cams:
            jsons_mask[parking_ground]["positions_mask"][cam] = parking_space_initializer.positions_mask[cam]
            jsons_mask[parking_ground]["square_of_mask"][cam] = parking_space_initializer.square_of_mask[cam]

            jsons_mask[parking_ground]["outlier_positions_mask"][cam] = parking_space_initializer.outlier_positions_mask[cam]
            jsons_mask[parking_ground]["outlier_square_of_mask"][cam] = parking_space_initializer.outlier_square_of_mask[cam]


    with open(os.path.join(ROOT_DIR, args.save_dir, "cells_mask_and_square_of_mask.json"), "w") as f:
        json.dump(jsons_mask, f, cls=NumpyEncoder, indent=4)
