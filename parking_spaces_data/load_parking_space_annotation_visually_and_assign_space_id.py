import sys
import os
from operator import itemgetter
from itertools import groupby
import argparse
import json
import numpy as np
import cv2

from parking_spaces_data.sa_order_in_json_to_unified_id import sa_order_in_json_to_unified_id
from parking_spaces_data.pa_order_in_json_to_unified_id import pa_order_in_json_to_unified_id


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=r"F:\Parking_Spaces_Recommendation_Data\PA_parking_spaces", help="Directory contains images")
    parser.add_argument("--label_file_path", type=str, default="pa_parking_spaces_annotation.json", help="Path to parking spaces annotation file")

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
        print("Image id: {}".format(image_id))

        for item in items:
            print("Path of image id {0} is {1}".format(image_id, item["file_name"]))
            file_name = item["file_name"]
            width = item["width"]
            height = item["height"]
            file_path = os.path.join(args.dataset_dir, file_name)

        parking_spaces = list(filter(lambda x: x["image_id"] == image_id, annotations))
        print("Parking spaces of image id {0} is {1}".format(image_id, len(parking_spaces)))

        img = cv2.imread(file_path)

        #for id, parking_space in groupby(parking_spaces, key=itemgetter("id")):
        #    #print("id: {0}, len: {1}".format(id, len(list(parking_space))))
        #    for space in parking_space:
        #        segm = space["segmentation"]
        space_id_list = []
        for i, parking_space in enumerate(parking_spaces):
            segmentation = parking_space["segmentation"]
            id = parking_space["id"]
            space_id_list.append(id)
            segmentation = np.array(segmentation, dtype=np.uint16).reshape(-1, 2)
            center_x, center_y = np.mean(segmentation, axis=0).astype(np.uint16)
            segmentation = segmentation.tolist()
            color = (np.random.randint(150, 255), np.random.randint(150, 255), np.random.randint(150, 255))
            for j, point in enumerate(segmentation):
                x1, y1, = point
                if j < len(segmentation) - 1:
                    x2, y2 = segmentation[j + 1]
                else:
                    x2, y2 = segmentation[0]

                cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=2)
            #cv2.putText(img, "{}".format(sa_order_in_json_to_unified_id[id]), (center_x, center_y),
            #            cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), thickness=1)
            cv2.putText(img, "{}".format(id), (center_x, center_y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), thickness=1)
        print("Space id list {} in the image id {}".format(space_id_list, image_id))
        cv2.imshow("Image id {}".format(image_id), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    root_dir = os.path.abspath(".")
    sys.path.append(root_dir)

    args = get_args()

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)
