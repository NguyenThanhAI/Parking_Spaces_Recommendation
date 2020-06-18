import os
import argparse
from itertools import groupby
from operator import itemgetter
import json
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_file_path", type=str, default=r"D:\augmented_label\Augmented_Label.json", help="Path to label file")
    parser.add_argument("--fraction", type=float, default=0.7, help="Fraction of training and testing dataset")

    args = parser.parse_args()

    return args


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


def parse_json_label(args, json_label):
    assert isinstance(json_label, dict)

    images = json_label["images"]
    annotations = json_label["annotations"]
    categories = json_label["categories"]
    info = json_label["info"]
    licenses = json_label["licenses"]

    assert isinstance(images, list) and isinstance(annotations, list)

    training_images = []
    training_annotations = []

    testing_images = []
    testing_annotations = []

    training_img_id = 0
    training_anno_id = 0

    testing_img_id = 0
    testing_anno_id = 0

    for image_id, items in groupby(sorted(annotations, key=itemgetter("image_id")), key=itemgetter("image_id")):

        image_info = list(filter(lambda x: x["id"] == image_id, images))
        assert len(image_info) == 1

        prob = np.random.rand()

        if prob < args.fraction:

            image_info = image_info[0]

            image_info["id"] = training_img_id

            for item in items:

                item["image_id"] = training_img_id
                item["id"] = training_anno_id
                training_annotations.append(item)
                training_anno_id += 1

            training_images.append(image_info)
            #training_annotations.extend(list(items))

            training_img_id += 1
        else:

            image_info = image_info[0]

            image_info["id"] = testing_img_id

            for item in items:
                item["image_id"] = testing_img_id
                item["id"] = testing_anno_id
                testing_annotations.append(item)
                testing_anno_id += 1

            testing_images.append(image_info)
            #testing_annotations.extend(list(items))

            testing_img_id += 1

    training_json = {}
    #print(training_annotations)
    training_json["images"] = training_images
    training_json["annotations"] = training_annotations
    training_json["categories"] = categories
    training_json["info"] = info
    training_json["licenses"] = licenses

    testing_json = {}
    #print(testing_annotations)
    testing_json["images"] = testing_images
    testing_json["annotations"] = testing_annotations
    testing_json["categories"] = categories
    testing_json["info"] = info
    testing_json["licenses"] = licenses

    with open(os.path.join(os.path.dirname(args.label_file_path), "Training_" + os.path.basename(args.label_file_path)), "w") as f:
        json.dump(training_json, f)

    with open(os.path.join(os.path.dirname(args.label_file_path), "Testing_" + os.path.basename(args.label_file_path)), "w") as f:
        json.dump(testing_json, f)


if __name__ == '__main__':
    args = get_args()

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)
