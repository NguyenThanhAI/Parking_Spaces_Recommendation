import os
import argparse
from itertools import groupby
from operator import itemgetter
import json
import numpy as np
import cv2
from skimage.transform import rescale
from skimage.draw import polygon



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=r"D:\COCO_dataset\COCO_2017\train", help="Directory of dataset")
    parser.add_argument("--label_file_path", type=str, default=r"D:\COCO_dataset\annotations_trainval2017\annotations\truncated_train2017.json", help="Path to label file")

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

    assert isinstance(images, list) and isinstance(annotations, list)

    target_size_range = [20, 150]

    for image_id, items in groupby(sorted(images, key=lambda x: x["id"]), key=lambda x: x["id"]):
        for item in items:
            #print("Path of image of image id {0} is {1}".format(image_id, item["file_name"]))
            file_name = item["file_name"]
            width = item["width"]
            height = item["height"]
            file_path = os.path.join(args.dataset_dir, file_name)

        cars_or_vehicles = list(filter(lambda x: x["image_id"] == image_id and (x["area"] > 70000.), annotations))

        img = cv2.imread(file_path)

        for i, car_or_vehicle in enumerate(cars_or_vehicles):
            bbox = car_or_vehicle["bbox"]
            segmentation = car_or_vehicle["segmentation"]
            area = car_or_vehicle["area"]
            if not isinstance(segmentation, list):
                continue
            if len(segmentation) != 1:
                continue

            segmentation = np.array(segmentation, dtype=np.uint16).reshape(-1, 2)
            x_min, y_min = np.min(segmentation, axis=0)
            x_max, y_max = np.max(segmentation, axis=0)
            segmentation = segmentation - np.array([x_min, y_min])[np.newaxis, :]

            cropped_instance = img[y_min: y_max + 1, x_min: x_max + 1].copy()

            target_size = np.random.randint(low=target_size_range[0], high=target_size_range[1] + 1)

            print(target_size)

            scale_factor = target_size / cropped_instance.shape[0]

            segmentation = (segmentation * scale_factor).astype(int)

            area = area * (scale_factor ** 2)

            cc, rr = segmentation.T

            rr, cc = polygon(rr, cc)

            cropped_instance = rescale(cropped_instance, scale=scale_factor)

            mask = np.zeros_like(cropped_instance, dtype=np.bool)

            mask[rr, cc] = True

            cropped_instance = np.where(mask, cropped_instance, np.zeros_like(cropped_instance))

            for j, point in enumerate(segmentation):
                x1, y1 = point
                if j < len(segmentation) - 1:
                    x2, y2 = segmentation[j + 1]
                else:
                    x2, y2 = segmentation[0]
                cv2.line(cropped_instance, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)
            x, y, w, h = list(map(lambda x: int(x), bbox))
            x = int((x - x_min) * scale_factor)
            y = int((y - y_min) * scale_factor)
            w = int(w * scale_factor)
            h = int(h * scale_factor)
            cv2.rectangle(cropped_instance, (x, y), (x + w, y + h), color=(255, 255, 0), thickness=0)

            cv2.imshow("Image id {0} instance {1}".format(image_id, i), cropped_instance)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    args = get_args()

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)