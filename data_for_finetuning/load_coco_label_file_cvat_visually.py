import os
from operator import itemgetter
from itertools import groupby
import argparse
import json
import numpy as np
import cv2
import shapely


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=r"F:\Parking_Spaces_Recommendation_Data\finetuning_dataset", help="Directory of dataset")
    parser.add_argument("--label_file_path", type=str, default=r"C:\Users\Thanh_Tuyet\Downloads\Label_car.json", help="Path to label file")

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

    for image_id, items in groupby(images, key=itemgetter("id")):
        print("image_id:", image_id)
        for item in items:
            print("Path of image of image id {0} is {1}".format(image_id, item["file_name"]))
            file_name = item["file_name"]
            width = item["width"]
            height = item["height"]
            file_path = os.path.join(args.dataset_dir, file_name)

        cars_or_vehicles = list(filter(lambda x: x["image_id"] == image_id, annotations))
        print("Cars or vehicles of image id {0} is {1}".format(image_id, len(cars_or_vehicles)))

        img = cv2.imread(file_path)

        for i, car_or_vehicle in enumerate(cars_or_vehicles):
            bbox = car_or_vehicle["bbox"]
            segmentation = car_or_vehicle["segmentation"]
            segmentation = np.array(segmentation, dtype=np.uint16).reshape(-1, 2).tolist()
            for j, point in enumerate(segmentation):
                x1, y1 = point
                if j < len(segmentation) - 1:
                    x2, y2 = segmentation[j + 1]
                else:
                    x2, y2 = segmentation[0]
                cv2.line(img, (x1, y1), (x2, y2), color=(0, 255, 255), thickness=1)
            x, y, w, h = list(map(lambda x: int(x), bbox))
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(255, 255, 0), thickness=0)

        cv2.imshow("Image id {0}".format(image_id), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':

    args = get_args()

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)
