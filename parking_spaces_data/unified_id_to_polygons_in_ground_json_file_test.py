import sys
import os
from operator import itemgetter
from itertools import groupby
import argparse
import json
import numpy as np
from skimage.draw import polygon
import cv2

from parking_spaces_data.sa_pa_parking_spaces_ground_annotation import sa_in_order_in_json_to_unified_id, pa_in_order_in_json_to_unified_id


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_ground_dir", type=str, default=r"C:\Users\Thanh\Downloads\Parking_ground_images_and_db", help="Directory contains images")
    parser.add_argument("--label_file_path", type=str, default="parking_spaces_unified_id_segmen_in_ground.json", help="Path to parking spaces annotation file")
    parser.add_argument("--image_save_dir", type=str, default=r"C:\Users\Thanh\Downloads\Parking_ground_images_and_db", help="Directory contains result images")

    args = parser.parse_args()

    return args


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


def parse_json_label(args, json_label, save_images=True):

    if save_images:
        if not os.path.exists(args.image_save_dir):
            os.makedirs(args.image_save_dir, exist_ok=True)

    unified_id_to_polygons = json_label

    for parking_ground in unified_id_to_polygons:
        if parking_ground == "parking_ground_SA":
            file_name = "cropped_SA.jpg"
        else:
            file_name = "cropped_PA.jpg"

        file_path = os.path.join(args.image_ground_dir, file_name)
        img = cv2.imread(file_path)

        for unified_id in unified_id_to_polygons[parking_ground]:
            segment = unified_id_to_polygons[parking_ground][unified_id]["positions"]
            segment = np.array(segment, dtype=np.uint16).reshape(-1, 2)
            cc, rr = segment.T
            rr, cc = polygon(rr, cc)
            img[rr, cc] = np.random.randint(0, 255, [3], dtype=np.uint8)
            center_x, center_y = np.mean(segment, axis=0).astype(np.uint16)
            segment = segment.tolist()
            color = (np.random.randint(150, 255), np.random.randint(150, 255), np.random.randint(150, 255))
            for j, point in enumerate(segment):
                x1, y1, = point
                if j < len(segment) - 1:
                    x2, y2 = segment[j + 1]
                else:
                    x2, y2 = segment[0]

                cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=1)
            #cv2.putText(img, "{}".format(sa_order_in_json_to_unified_id[id]), (center_x, center_y),
            #            cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), thickness=1)
            cv2.putText(img, "{}".format(unified_id), (center_x, center_y),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), thickness=1)
        cv2.imshow("{}".format(parking_ground), img)
        cv2.waitKey(0)
        if save_images:
            file_name = "".join([os.path.basename(file_path).split(".")[0], "_result.jpg"])
            cv2.imwrite(os.path.join(args.image_save_dir, file_name), img)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    root_dir = os.path.abspath(".")
    sys.path.append(root_dir)

    args = get_args()

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)
