import sys
import os

sys.path.append(os.path.abspath("."))

import json
import argparse
import numpy as np
from skimage.draw import polygon
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--json_file", type=str, default="parking_ground_cam_to_considered_area.json", help="Path to config considered area json file")

    args = parser.parse_args()

    return args


def read_json_file(json_file_path):
    with open(json_file_path, "r")  as f:
        unified_id_to_polygons = json.load(f)

    return unified_id_to_polygons


if __name__ == '__main__':
    args = get_args()

    parking_ground_cam_to_considered_area = read_json_file(json_file_path=args.json_file)

    for parking_ground in parking_ground_cam_to_considered_area:
        for cam in parking_ground_cam_to_considered_area[parking_ground]:
            image = np.ones(shape=[720, 1280, 3], dtype=np.uint8) * 255

            areas = parking_ground_cam_to_considered_area[parking_ground][cam]

            for i, area in enumerate(areas):
                area = np.array(area, dtype=np.uint16).reshape(-1, 2)
                cc, rr = area.T
                rr, cc = polygon(rr, cc)
                image[rr, cc] = np.random.randint(0, 255, [3], dtype=np.uint8)
                center_x, center_y = np.mean(area, axis=0).astype(np.uint16)
                area = area.tolist()

                color = (0, 0, 255)

                for j, point in enumerate(area):
                    x1, y1 = point
                    if j < len(area) - 1:
                        x2, y2 = area[j + 1]
                    else:
                        x2, y2 = area[0]
                    cv2.line(image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(image, str(i + 1), (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)

            cv2.imshow(parking_ground + "_" + cam, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()