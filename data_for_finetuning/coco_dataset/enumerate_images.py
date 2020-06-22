import os
import argparse
import asyncio
from itertools import groupby
from operator import itemgetter
import json
import requests


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_file_path", type=str, default=r"D:\COCO_dataset\annotations_trainval2017\annotations\instances_val2017.json", help="Path to label json file")
    parser.add_argument("--considered_classes", type=str, default="bicycle,motorcycle,car,bus,truck")
    parser.add_argument("--save_dir", type=str, default=r"D:\COCO_dataset\COCO_2017", help="Images directory")
    parser.add_argument("--output_label_path", type=str, default=r"truncated_val2017.json", help="Output label json")

    args = parser.parse_args()

    return args


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


def download_image_from_url(url, save_dir, name):
    response = requests.get(url)

    with open(os.path.join(save_dir, name), "wb") as f:
        f.write(response.content)
        f.close()


def parse_json_label(args, json_label):

    assert isinstance(json_label, dict)

    images = json_label["images"]
    annotations = json_label["annotations"]
    categories = json_label["categories"]
    info = json_label["info"]
    licenses = json_label["licenses"]

    output_images = []
    output_annotations = []
    output_categories = []

    assert isinstance(images, list) and isinstance(annotations, list) and isinstance(categories, list)

    considered_classes = args.considered_classes.split(",")

    coco_class_to_id_dict = {}

    class_to_final_class_id = {"bicycle": 4, "motorcycle": 4, "car": 1, "truck": 2, "bus": 3}
    final_class_id_to_class = {1: "Car", 2: "Truck", 3: "Bus", 4: "Bicycle"}

    for considered_class in considered_classes:
        category = list(filter(lambda x: x["name"] == considered_class, categories))
        #print(considered_class, category, len(category))
        assert len(category) == 1
        category = category[0]
        class_id = category["id"]
        coco_class_to_id_dict[considered_class] = class_id
        category["id"] = class_to_final_class_id[considered_class]
        category["name"] = final_class_id_to_class[class_to_final_class_id[considered_class]]
        category["supercategory"] = ""
        output_categories.append(category)
    print(output_categories)
    output_categories = [dict(y) for y in set(tuple(x.items()) for x in output_categories)]
    print(output_categories)

    coco_class_id_to_final_class_id = {coco_class_to_id_dict[k]: v for k, v in class_to_final_class_id.items()}

    annotations = list(filter(lambda x: x["category_id"] in list(coco_class_to_id_dict.values()), annotations))
    print(list(map(lambda x: x["category_id"], annotations)))
    img_id = 0
    anno_id = 0

    for image_id, items in groupby(sorted(annotations, key=itemgetter("image_id")), key=itemgetter("image_id")):
        image_info = list(filter(lambda x: x["id"] == image_id, images))
        assert len(image_info) == 1

        image_info = image_info[0]
        image_info["id"] = img_id
        output_images.append(image_info)
        download_image_from_url(url=image_info["coco_url"], save_dir=args.save_dir, name=image_info["file_name"])
        print("Dowwload image id {} name {} from link {}".format(img_id, image_info["file_name"], image_info["coco_url"]))

        for item in items:
            item["id"] = anno_id
            item["category_id"] = coco_class_id_to_final_class_id[item["category_id"]]
            item["image_id"] = img_id
            output_annotations.append(item)
            anno_id += 1

        img_id += 1

    output_json = {}
    output_json["images"] = output_images
    output_json["annotations"] = output_annotations
    output_json["categories"] = output_categories
    output_json["info"] = info
    output_json["licenses"] = licenses

    with open(os.path.join(os.path.dirname(args.label_file_path), args.output_label_path), "w") as f:
        json.dump(output_json, f)


if __name__ == '__main__':
    args = get_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)
