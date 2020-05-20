import sys
import os
from operator import itemgetter
from itertools import groupby
import argparse
import json

from parking_spaces_data.sa_pa_parking_spaces_ground_annotation import sa_in_order_in_json_to_unified_id, pa_in_order_in_json_to_unified_id
from parking_spaces_data.sa_order_in_json_to_unified_id import sa_type_space_to_unified_id
from parking_spaces_data.pa_order_in_json_to_unified_id import pa_type_space_to_unified_id


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sa_pa_label_file_path", type=str, default="parking_spaces_ground_annotation.json", help="Path to parking spaces annotation file")

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

    unified_id_to_polygons = {}

    json_label = read_label_file(label_file_path=args.sa_pa_label_file_path)

    assert isinstance(json_label, dict)
    images = json_label["images"]
    annotations = json_label["annotations"]

    assert isinstance(images, list) and isinstance(annotations, list)

    for image_id, items in groupby(images, key=itemgetter("id")):
        print("Image id: {}".format(image_id))

        if image_id == 0:
            order_in_json_to_unified_id = pa_in_order_in_json_to_unified_id
            type_space_to_unified_id = pa_type_space_to_unified_id
            parking_ground = "parking_ground_PA"
            unified_id_to_polygons[parking_ground] = {}
        else:
            order_in_json_to_unified_id = sa_in_order_in_json_to_unified_id
            type_space_to_unified_id = sa_type_space_to_unified_id
            parking_ground = "parking_ground_SA"
            unified_id_to_polygons[parking_ground] = {}

        parking_spaces = list(filter(lambda x: x["image_id"] == image_id, annotations))
        print("Parking spaces of image id {0} is {1}".format(image_id, len(parking_spaces)))

        unified_id_to_type_space = {}
        for k, v in type_space_to_unified_id.items():
            unified_id_to_type_space.update(dict(zip(v, [k] * len(v))))

        for i, parking_space in enumerate(parking_spaces):
            segmentation = parking_space["segmentation"]
            id = parking_space["id"]
            if id in order_in_json_to_unified_id:
                unified_id = order_in_json_to_unified_id[id]
            else:
                continue

            if unified_id not in unified_id_to_polygons[parking_ground]:
                unified_id_to_polygons[parking_ground][unified_id] = {}

            if "positions" not in unified_id_to_polygons[parking_ground][unified_id]:
                unified_id_to_polygons[parking_ground][unified_id]["positions"] = segmentation

            if "type_space" not in unified_id_to_polygons[parking_ground][unified_id]:
                unified_id_to_polygons[parking_ground][unified_id]["type_space"] = unified_id_to_type_space[unified_id]

    with open("parking_spaces_unified_id_segmen_in_ground.json", "w") as f:
        json.dump(unified_id_to_polygons, f, indent=5)