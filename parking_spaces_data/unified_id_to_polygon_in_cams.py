import sys
import os
import argparse
import json

from parking_spaces_data.order_in_json_to_unified_id import order_in_json_to_unified_id, cam_to_space_id


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_file_path", type=str, default="parking_spaces_annotation.json", help="Path to parking spaces annotation file")

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

    json_label = read_label_file(label_file_path=args.label_file_path)

    annotations = json_label["annotations"]

    space_id_to_cam = {}
    for k, v in cam_to_space_id.items():
        space_id_to_cam.update(dict(zip(v, [k] * len(v))))

    unified_id_to_polygons = {}
    for order in order_in_json_to_unified_id:
        unified_id = order_in_json_to_unified_id[order]
        cam = space_id_to_cam[order]

        if unified_id not in unified_id_to_polygons:
            unified_id_to_polygons[unified_id] = {}

        if "positions" not in unified_id_to_polygons[unified_id]:
            unified_id_to_polygons[unified_id]["positions"] = {}

        annotation_wrt_order = list(filter(lambda x: x["id"] == order, annotations))
        assert len(annotation_wrt_order) == 1

        unified_id_to_polygons[unified_id]["positions"][cam] = annotation_wrt_order[0]["segmentation"]

    with open("parking_spaces_unified_id_segmen_in_cameras.json", "w") as f:
        json.dump(unified_id_to_polygons, f, indent=4)
