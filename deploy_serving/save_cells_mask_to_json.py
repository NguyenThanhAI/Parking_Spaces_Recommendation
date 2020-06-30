import sys
import os
import argparse
import codecs, json
import numpy as np

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

from parking_spaces_assignment.parking_space import ParkingSpacesInitializer


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

        parking_space_initializer = ParkingSpacesInitializer(active_cams=active_cams,
                                                             parking_ground=parking_ground,
                                                             shape=shape,
                                                             config_json_path=config_json_path)

        parking_spaces_list, outlier_parking_spaces_list = parking_space_initializer.initialize_parking_spaces()

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
