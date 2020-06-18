import os
import argparse
from itertools import groupby
from operator import itemgetter
import json
import numpy as np
from skimage.draw import polygon
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--landscape_dir", type=str, default=r"C:\Users\Thanh\Downloads\1280x720_landscape", help="Path to landscape images directory")
    parser.add_argument("--original_dir", type=str, default=r"D:\finetuning_dataset", help="Path to original images dataset")
    parser.add_argument("--label_file_path", type=str, default=r"C:\Users\Thanh\Downloads\Annotations\unzip\Label.json", help="Annotated json file")
    parser.add_argument("--output_landscape_dir", type=str, default=r"D:\augmented_finetuning_dataset", help="Output dir")
    parser.add_argument("--output_label_file", type=str, default=r"D:\augmented_label\Augmented_Label.json", help="Path to output label file path")

    args = parser.parse_args()

    return args


def enumerate_landscape_images(dir):
    images_list = []
    for dirs, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".jpg"):
                images_list.append(os.path.join(dirs, file))

    return images_list


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


def parse_json_label(args, json_label):
    landscape_list = enumerate_landscape_images(args.landscape_dir)

    assert isinstance(json_label, dict)

    images = json_label["images"]
    annotations = json_label["annotations"]
    categories = json_label["categories"]
    info = json_label["info"]
    licenses = json_label["licenses"]

    assert isinstance(images, list) and isinstance(annotations, list)

    accumulated_images = []
    accumulated_annotations = []
    img_id = 0
    anno_id = 0

    augmented_image_id = 0

    for image_id, items in groupby(sorted(annotations, key=itemgetter("image_id")), key=itemgetter("image_id")):

        mask = np.zeros(shape=[720, 1280], dtype=np.bool)

        image_info = list(filter(lambda x: x["id"] == image_id, images))
        assert len(image_info) == 1
        image_path = image_info[0]["file_name"]

        original_image_info = image_info[0].copy() # Phải [0] trước copy
        augmented_image_info = image_info[0].copy()

        original_file_name = "original_" + str(augmented_image_id) + ".jpg"
        augmented_file_name = "augmented_" + str(augmented_image_id) + ".jpg"

        original_image_info["id"] = img_id
        original_image_info["file_name"] = original_file_name

        img_id += 1

        augmented_image_info["id"] = img_id
        augmented_image_info["file_name"] = augmented_file_name

        img_id += 1

        accumulated_images.extend([original_image_info, augmented_image_info])

        image = cv2.imread(os.path.join(args.original_dir, image_path))

        landscape = np.random.choice(landscape_list)
        print(landscape)
        landscape_image = cv2.imread(landscape)
        if landscape_image.shape[0] != 720 or landscape_image.shape[1] != 1280:
            landscape_image = cv2.resize(landscape_image, (1280, 720))

        for item in items:
            segmentation = item["segmentation"]
            segmentation = np.array(segmentation, dtype=np.uint16).reshape(-1, 2)
            cc, rr = segmentation.T
            rr, cc = polygon(rr, cc)
            mask[rr, cc] = True
            original_item = item.copy()
            original_item["id"] = anno_id
            original_item["image_id"] = img_id - 2

            anno_id += 1

            augmented_item = item.copy()
            augmented_item["id"] = anno_id
            augmented_item["image_id"] = img_id - 1

            anno_id += 1

            accumulated_annotations.extend([original_item, augmented_item])

        #image = np.where(mask[:, :, np.newaxis], image, np.zeros_like(image, dtype=np.uint8))
        landscape_image = np.where(mask[:, :, np.newaxis], image, landscape_image)
        cv2.imwrite(os.path.join(args.output_landscape_dir, augmented_file_name), landscape_image)
        cv2.imwrite(os.path.join(args.output_landscape_dir, original_file_name), image)
        augmented_image_id += 1
        cv2.imshow("Anh", image)
        cv2.imshow("Landscape", landscape_image)
        cv2.waitKey(500)

    accumulated_json = {}

    accumulated_json["images"] = accumulated_images
    accumulated_json["annotations"] = accumulated_annotations
    accumulated_json["licenses"] = licenses
    accumulated_json["categories"] = categories
    accumulated_json["info"] = info

    if not os.path.exists(os.path.dirname(args.output_label_file)):
        os.makedirs(os.path.dirname(args.output_label_file), exist_ok=True)

    with open(args.output_label_file, "w") as f:
        json.dump(accumulated_json, f)


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.output_landscape_dir):
        os.makedirs(args.output_landscape_dir, exist_ok=True)

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)