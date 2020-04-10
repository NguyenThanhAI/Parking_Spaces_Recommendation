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

    parser.add_argument("--json_file", type=str, default="parking_spaces_unified_id_segmen_in_cameras.json", help="Path to unified id to polygons in cams json file")

    args = parser.parse_args()

    return args


def read_json_file(json_file_path):
    with open(json_file_path, "r")  as f:
        unified_id_to_polygons = json.load(f)

    return unified_id_to_polygons


if __name__ == '__main__':
    args = get_args()

    unified_id_to_polygons = read_json_file(json_file_path=args.json_file)

    for cam in ["cam_1", "cam_2", "cam_3"]:
        image = np.ones(shape=[720, 1280, 3], dtype=np.uint8) * 255

        for unified_id in unified_id_to_polygons:
            if cam in unified_id_to_polygons[unified_id]["positions"]:
                segment = unified_id_to_polygons[unified_id]["positions"][cam]
                segment = np.array(segment, dtype=np.uint16).reshape(-1, 2)
                cc, rr = segment.T
                rr, cc = polygon(rr, cc)
                image[rr, cc] = np.random.randint(0, 255, [3], dtype=np.uint8)
                center_x, center_y = np.mean(segment, axis=0).astype(np.uint16)
                segment = segment.tolist()

                color = (0, 0, 255)

                for j, point in enumerate(segment):
                    x1, y1 = point
                    if j < len(segment) - 1:
                        x2, y2 = segment[j + 1]
                    else:
                        x2, y2 = segment[0]
                    cv2.line(image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(image, unified_id, (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 1)
        cv2.imshow(cam, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
