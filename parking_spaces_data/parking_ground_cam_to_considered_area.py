import sys
import os
import json
from itertools import groupby
from operator import itemgetter
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_file_path", type=str, default=r"C:\Users\Thanh\Downloads\Config_Considered_Area\annotations\instances_default.json", help="Annotation file")

    args = parser.parse_args()

    return args


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


if __name__ == '__main__':
    root_dir = os.path.abspath(".")
    sys.path.append(root_dir)

    args = get_args()

    json_label = read_label_file(args.label_file_path)

    parking_ground_cam_to_considered_area = {}

    images = json_label["images"]
    annotations = json_label["annotations"]

    assert isinstance(images, list) and isinstance(annotations, list)

    for image_id, items in groupby(images, key=itemgetter("id")):

        for item in items:
            print("Path of image id {0} is {1}".format(image_id, item["file_name"]))
            file_name = item["file_name"]
            width = item["width"]
            height = item["height"]

        considered_areas = list(filter(lambda x: x["image_id"] == image_id, annotations))

        if file_name.split(".")[0].startswith("SA"):
            parking_ground = "parking_ground_SA"
        else:
            parking_ground = "parking_ground_PA"

        if file_name.split(".")[0].endswith("cam_1"):
            cam = "cam_1"
        elif file_name.split(".")[0].endswith("cam_2"):
            cam = "cam_2"
        elif file_name.split(".")[0].endswith("cam_3"):
            cam = "cam_3"
        else:
            cam = "cam_4"

        if parking_ground not in parking_ground_cam_to_considered_area:
            parking_ground_cam_to_considered_area[parking_ground] = {}

        if cam not in parking_ground_cam_to_considered_area[parking_ground]:
            parking_ground_cam_to_considered_area[parking_ground][cam] = []

        for considered_area in considered_areas:
            segment = considered_area["segmentation"]
            parking_ground_cam_to_considered_area[parking_ground][cam].append(segment)

    with open("parking_ground_cam_to_considered_area.json", "w") as f:
        json.dump(parking_ground_cam_to_considered_area, f, indent=3)

