import os
import argparse
import shutil
from itertools import groupby
from operator import itemgetter
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_file_list", type=str, required=True, help="Label files list")
    parser.add_argument("--dataset_dir_list", type=str, required=True, help="Dataset directories list")
    parser.add_argument("--output_image_dir", type=str, default=r"D:\COCO_dataset\COCO_Masks", help="Save image directory")
    parser.add_argument("--output_label_file_dir", type=str, default=r"D:\COCO_dataset\annotation_masks", help="Output directory of json file")
    parser.add_argument("--output_label_file", type=str, default="=coco_instances_mask.json", help="Output file of json")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.output_label_file_dir):
        os.makedirs(args.output_label_file_dir)

    if not os.path.exists(args.output_image_dir):
        os.makedirs(args.output_image_dir)

    label_file_list = args.label_file_list.split(",")
    dataset_dir_list = args.dataset_dir_list.split(",")

    assert len(label_file_list) == len(dataset_dir_list)

    accumulated_images = []
    accumulated_annotations = []
    accumulated_categories = []
    info = []
    licenses = []

    for json_file, dataset_dir in zip(label_file_list, dataset_dir_list):

        with open(json_file, "r") as f:
            json_label = json.load(f)

        assert isinstance(json_label, dict)

        images = json_label["images"]
        annotations = json_label["annotations"]
        categories = json_label["categories"]
        info = json_label["info"]
        licenses = json_label["licenses"]

        nrof_previous_images = len(accumulated_images)
        nrof_previous_annotations = len(accumulated_annotations)

        img_id = 0
        anno_id = 0

        image_id_dict = {}  # Convert old image id to new image id

        for image_id, items in groupby(sorted(annotations, key=itemgetter("image_id")), key=itemgetter("image_id")):
            print("json file: {}, dataset dir: {}, image_id: {}".format(json_file, dataset_dir, image_id))
            image_id_dict[image_id] = img_id + nrof_previous_images
            for item in items:
                print("json file: {}, dataset dir: {}, image id: {}, item id: {}, new image id: {}, new item id: {}".format(json_file, dataset_dir, item["image_id"], item["id"], img_id + nrof_previous_images, anno_id + nrof_previous_annotations))
                item["image_id"] = img_id + nrof_previous_images
                item["id"] = anno_id + nrof_previous_annotations
                anno_id += 1
            img_id += 1

        images = list(filter(lambda x: x["id"] in list(image_id_dict.keys()), images))

        for image in images:
            image["id"] = image_id_dict[image["id"]]
            file_name = str(image["id"]) + ".jpg"
            shutil.copy(os.path.join(dataset_dir, image["file_name"]), os.path.join(args.output_image_dir, file_name))
            image["file_name"] = file_name

        accumulated_images.extend(images)
        accumulated_annotations.extend(annotations)
        accumulated_categories.extend(categories)

    accumulated_categories = [dict(y) for y in set(tuple(x.items()) for x in accumulated_categories)]

    accumulated_json = {}

    accumulated_json["images"] = accumulated_images
    accumulated_json["annotations"] = accumulated_annotations
    accumulated_json["licenses"] = licenses
    accumulated_json["categories"] = accumulated_categories
    accumulated_json["info"] = info

    with open(os.path.join(args.output_label_file_dir, args.output_label_file), "w") as f:
        json.dump(accumulated_json, f)
