import sys
import os
import argparse
import json

from parking_spaces_data.sa_order_in_json_to_unified_id import sa_order_in_json_to_unified_id, sa_cam_to_space_id, sa_unified_id_and_adjacency_ids, sa_unified_id_to_orientation_consideration, sa_type_space_to_unified_id, sa_cam_to_considered_unified_id
from parking_spaces_data.pa_order_in_json_to_unified_id import pa_order_in_json_to_unified_id, pa_cam_to_space_id, pa_unified_id_and_adjacency_ids, pa_unified_id_to_orientation_consideration, pa_type_space_to_unified_id, pa_cam_to_considered_unified_id


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sa_label_file_path", type=str, default="sa_parking_spaces_annotation.json", help="Path to SA parking spaces annotation file")
    parser.add_argument("--pa_label_file_path", type=str, default="pa_parking_spaces_annotation.json", help="Path to PS parking spaces annotation file")

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
    for parking_ground in ["parking_ground_SA", "parking_ground_PA"]:

        if parking_ground == "parking_ground_SA":

            json_label = read_label_file(label_file_path=args.sa_label_file_path)
            cam_to_space_id = sa_cam_to_space_id
            order_in_json_to_unified_id = sa_order_in_json_to_unified_id
            unified_id_and_adjacency_ids = sa_unified_id_and_adjacency_ids
            unified_id_to_orientation_consideration = sa_unified_id_to_orientation_consideration
            type_space_to_unified_id = sa_type_space_to_unified_id
            cam_to_considered_unified_id = sa_cam_to_considered_unified_id

        else:

            json_label = read_label_file(label_file_path=args.pa_label_file_path)
            cam_to_space_id = pa_cam_to_space_id
            order_in_json_to_unified_id = pa_order_in_json_to_unified_id
            unified_id_and_adjacency_ids = pa_unified_id_and_adjacency_ids
            unified_id_to_orientation_consideration = pa_unified_id_to_orientation_consideration
            type_space_to_unified_id = pa_type_space_to_unified_id
            cam_to_considered_unified_id = pa_cam_to_considered_unified_id

        annotations = json_label["annotations"]

        space_id_to_cam = {}
        for k, v in cam_to_space_id.items():
            space_id_to_cam.update(dict(zip(v, [k] * len(v))))

        unified_id_to_type_space = {}
        for k, v in type_space_to_unified_id.items():
            unified_id_to_type_space.update(dict(zip(v, [k] * len(v))))

        considered_unified_id_to_cam = {}
        for k, v in cam_to_considered_unified_id.items():
            considered_unified_id_to_cam.update(dict(zip(v, [k] * len(v))))

        unified_id_to_polygons[parking_ground] = {}
        for order in order_in_json_to_unified_id:
            unified_id = order_in_json_to_unified_id[order]
            cam = space_id_to_cam[order]

            if unified_id not in unified_id_to_polygons[parking_ground]:
                unified_id_to_polygons[parking_ground][unified_id] = {}

            if "positions" not in unified_id_to_polygons[parking_ground][unified_id]:
                unified_id_to_polygons[parking_ground][unified_id]["positions"] = {}

            if "reversed_considered_orients" not in unified_id_to_polygons[parking_ground][unified_id]:
                unified_id_to_polygons[parking_ground][unified_id]["reversed_considered_orients"] = {}

            annotation_wrt_order = list(filter(lambda x: x["id"] == order, annotations))
            assert len(annotation_wrt_order) == 1

            unified_id_to_polygons[parking_ground][unified_id]["positions"][cam] = annotation_wrt_order[0]["segmentation"]
            unified_id_to_polygons[parking_ground][unified_id].update(unified_id_and_adjacency_ids[unified_id])
            unified_id_to_polygons[parking_ground][unified_id]["reversed_considered_orients"] = unified_id_to_orientation_consideration[unified_id]
            unified_id_to_polygons[parking_ground][unified_id]["type_space"] = unified_id_to_type_space[unified_id]
            unified_id_to_polygons[parking_ground][unified_id]["considered_in_cam"] = considered_unified_id_to_cam[unified_id]

    with open("parking_spaces_unified_id_segmen_in_cameras.json", "w") as f:
        json.dump(unified_id_to_polygons, f, indent=5)
