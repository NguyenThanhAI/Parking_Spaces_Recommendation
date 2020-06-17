import os
import argparse
import zipfile
from itertools import groupby
from operator import itemgetter
import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_folder", type=str, default=r"C:\Users\Thanh\Downloads\Annotations", help="Path to annotation files")
    parser.add_argument("--result_folder", type=str, default=r"C:\Users\Thanh\Downloads\Annotations\unzip", help="Path to unzip folder")
    parser.add_argument("--result_json", type=str, default=r"Label.json", help="Path to unzip folder")

    args = parser.parse_args()

    return args


def enumerate_zip_files(dir):
    list_zip = []
    for dirs, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".zip"):
                list_zip.append(os.path.join(dir, file))

    return list_zip


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder, exist_ok=True)

    list_zip = enumerate_zip_files(args.dataset_folder)

    for zip in list_zip:
        with zipfile.ZipFile(zip, "r") as zip_ref:
            zip_ref.extractall(os.path.join(args.result_folder, zip.split(".")[0].split("_")[2]))

    members = sorted(list(map(lambda x: int(x), os.listdir(args.result_folder))))

    accumulated_images = []
    accumulated_annotations = []
    categories = []
    info = []
    licenses = []

    for member in members:
        json_file = os.path.join(args.result_folder, str(member), "annotations", "instances_default.json")

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

        image_id_dict = {} # Convert old image id to new image id

        for image_id, items in groupby(sorted(annotations, key=itemgetter("image_id")), key=itemgetter("image_id")):
            print("member: {}, image_id: {}".format(member, image_id))
            image_id_dict[image_id] = img_id + nrof_previous_images
            for item in items:
                print("member: {}, image id: {}, item id: {}, new image id: {}, new item id: {}".format(member, item["image_id"], item["id"], img_id + nrof_previous_images, anno_id + nrof_previous_annotations))
                item["image_id"] = img_id + nrof_previous_images
                item["id"] = anno_id + nrof_previous_annotations
                anno_id += 1
            img_id += 1

        images = list(filter(lambda x: x["id"] in list(image_id_dict.keys()), images))

        for image in images:
            image["id"] = image_id_dict[image["id"]]

        accumulated_images.extend(images)
        accumulated_annotations.extend(annotations)

    accumulated_json = {}

    accumulated_json["images"] = accumulated_images
    accumulated_json["annotations"] = accumulated_annotations
    accumulated_json["licenses"] = licenses
    accumulated_json["categories"] = categories
    accumulated_json["info"] = info

    with open(os.path.join(args.result_folder, args.result_json), "w") as f:
        json.dump(accumulated_json, f)