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
    print(list_zip)

    for zip in list_zip:
        with zipfile.ZipFile(zip, "r") as zip_ref:
            zip_ref.extractall(os.path.join(args.result_folder, zip.split(".")[0].split("_")[2]))

    members = sorted(list(map(lambda x: int(x), os.listdir(args.result_folder))))

    nrof_annotated_images = 0
    nrof_annotated_vehicles = 0

    for member in members:
        json_file = os.path.join(args.result_folder, str(member), "annotations", "instances_default.json")

        with open(json_file, "r") as f:
            json_label = json.load(f)

        assert isinstance(json_label, dict)

        images = json_label["images"]
        annotations = json_label["annotations"]

        nrof_annotated_vehicles += len(annotations)

        assert isinstance(images, list) and isinstance(annotations, list)

        for image_id, items in groupby(sorted(annotations, key=itemgetter("image_id")), key=itemgetter("image_id")):
            print("Image id {} of member {}".format(image_id, member))
            nrof_annotated_images += 1
        #print(len(groupby(sorted(annotations, key=itemgetter("image_id")), key=itemgetter("image_id"))))

    print("Number of annotated images is {}, number of annotated vehicles is {}".format(nrof_annotated_images, nrof_annotated_vehicles))