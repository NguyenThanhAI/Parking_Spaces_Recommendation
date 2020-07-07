import os
import argparse
from itertools import groupby, chain
from operator import itemgetter
import json
import numpy as np
import cv2
from skimage.draw import polygon


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label_file_path", type=str, default=r"D:\COCO_dataset\annotations_trainval2017\annotations\truncated_train2017.json", help="Path to label json file")
    parser.add_argument("--dataset_dir", type=str, default=r"D:\COCO_dataset\COCO_2017\train", help="Directory of dataset")
    parser.add_argument("--save_dir", type=str, default=r"D:\COCO_dataset\COCO_2017_Masks", help="Images directory")
    parser.add_argument("--output_label_path", type=str, default=r"truncated_train_instances_mask2017.json", help="Output label json")
    parser.add_argument("--min_area", type=int, default=50000, help="Min area of instance to be considered")
    parser.add_argument("--min_width", type=int, default=400, help="Min width of instance to be considered")
    parser.add_argument("--min_height", type=int, default=400, help="Min height of instance to be considered")

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

    chosen_images = []
    chosen_annotations = []

    img_id = 0
    anno_id = 0

    stopped = False

    for image_id, items in groupby(sorted(images, key=lambda x: x["id"]), key=lambda x: x["id"]):
        for item in items:
            # print("Path of image of image id {0} is {1}".format(image_id, item["file_name"]))
            file_name = item["file_name"]
            width = item["width"]
            height = item["height"]
            file_path = os.path.join(args.dataset_dir, file_name)

        if stopped:
            break

        cars_or_vehicles = list(filter(lambda x: x["image_id"] == image_id and x["area"] > args.min_area and x["bbox"][2] > args.min_width and x["bbox"][3] > args.min_height, annotations))

        img = cv2.imread(file_path)

        for i, car_or_vehicle in enumerate(cars_or_vehicles):

            if stopped:
                break

            bbox = car_or_vehicle["bbox"]
            segmentation = car_or_vehicle["segmentation"]
            area = car_or_vehicle["area"]

            if not isinstance(segmentation, list):
                continue

            if len(segmentation) != 1:
                continue

            x, y, w, h = list(map(lambda x: int(x), bbox))

            if area <= args.min_area or w <= args.min_width or h <= args.min_height:
                continue

            segmentation = np.array(segmentation, dtype=np.uint16).reshape(-1, 2)
            x_min, y_min = np.min(segmentation, axis=0)
            x_max, y_max = np.max(segmentation, axis=0)
            segmentation = segmentation - np.array([x_min, y_min])[np.newaxis, :]

            cropped_instance = img[y_min: y_max + 1, x_min: x_max + 1].copy()

            print("Image id: {}, item id: {}, instance shape: {}".format(image_id, i, cropped_instance.shape))

            cc, rr = segmentation.T

            rr, cc = polygon(rr, cc)

            mask = np.zeros_like(cropped_instance, dtype=np.bool)

            mask[rr, cc] = True

            cropped_instance = np.where(mask, cropped_instance, np.zeros_like(cropped_instance))

            cv2.imshow("Image id {0} instance {1}".format(image_id, i), cropped_instance)
            key = cv2.waitKey(0)
            if key == ord("s"):
                image_name = str(img_id) + ".jpg"
                height, width = cropped_instance.shape[:2]
                cc, rr = segmentation.T
                cc = cc.tolist()
                rr = rr.tolist()
                segmentation = [list(chain(*list(zip(cc, rr))))]

                bbox = [0, 0, width, height]


                car_or_vehicle["id"] = anno_id
                car_or_vehicle["image_id"] = img_id
                car_or_vehicle["segmentation"] = segmentation
                car_or_vehicle["bbox"] = bbox
                car_or_vehicle["area"] = area

                chosen_annotations.append(car_or_vehicle)
                chosen_images.append({"width": width, "license": 0, "date_captured": 0, "id": img_id, "coco_url": "",
                                      "file_name": image_name, "height": height, "flickr_url": ""})

                anno_id += 1
                img_id += 1
                cv2.imwrite(os.path.join(args.save_dir, image_name), cropped_instance)
            elif key == ord("q"):
                stopped = True
            cv2.destroyAllWindows()

    output_json = {}

    output_json["images"] = chosen_images
    output_json["annotations"] = chosen_annotations
    output_json["licenses"] = licenses
    output_json["categories"] = categories
    output_json["info"] = info

    with open(os.path.join(os.path.dirname(args.label_file_path), args.output_label_path), "w") as f:
        json.dump(output_json, f)


if __name__ == '__main__':
    args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)
