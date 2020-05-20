import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
from skimage.draw import polygon
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_ground_dir", type=str, default=r"C:\Users\Thanh\Downloads\Parking_ground_images_and_db", help="Directory contains images")
    parser.add_argument("--label_file_path", type=str, default="../parking_spaces_data/parking_spaces_unified_id_segmen_in_ground.json", help="Path to parking spaces annotation file")
    parser.add_argument("--image_save_dir", type=str, default=r"C:\Users\Thanh\Downloads\Parking_ground_images_and_db", help="Directory contains result images")
    parser.add_argument("--parking_ground", type=str, default="parking_ground_SA", help="Parking ground to visualize heatmap")
    parser.add_argument("--cells_heatmap_file", type=str, default="cells_heatmap.csv", help="Path to cells heatmap file")

    args = parser.parse_args()

    return args


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


def parse_json_label(args, json_label, save_images=True):
    color_dict = {0.: (255, 255, 255),
                  (0., 20.): (153, 51, 102),
                  (20., 40.): (255, 0, 0),
                  (40., 60.): (0, 255, 0),
                  (60., 80.): (0, 255, 255),
                  80.: (0, 0, 255)}
    if save_images:
        if not os.path.exists(args.image_save_dir):
            os.makedirs(args.image_save_dir, exist_ok=True)

    unified_id_to_polygons = json_label
    if args.parking_ground == "parking_ground_SA":
        file_name = "cropped_SA.jpg"
    else:
        file_name = "cropped_PA.jpg"

    file_path = os.path.join(args.image_ground_dir, file_name)
    img = cv2.imread(file_path)

    cells_heatmap = pd.read_csv(args.cells_heatmap_file)
    cells_heatmap = dict(zip(cells_heatmap.vehicle_id, cells_heatmap.heatmap))

    for unified_id in unified_id_to_polygons[args.parking_ground]:
        if int(unified_id) not in cells_heatmap:
            cells_heatmap[int(unified_id)] = 0.
        segment = unified_id_to_polygons[args.parking_ground][unified_id]["positions"]
        segment = np.array(segment, dtype=np.uint16).reshape(-1, 2)
        cc, rr = segment.T
        rr, cc = polygon(rr, cc)
        percent = cells_heatmap[int(unified_id)]
        for percent_time in color_dict:
            if isinstance(percent_time, float):
                if percent_time == 0.:
                    if percent <= percent_time:
                        color = color_dict[percent_time]
                        break
                    else:
                        continue
                else:
                    if percent > percent_time:
                        color = color_dict[percent_time]
                        break
                    else:
                        continue
            else:
                if percent > percent_time[0] and percent <= percent_time[1]:
                    color = color_dict[percent_time]
                    break
                else:
                    continue
        img[rr, cc] = color
        center_x, center_y = np.mean(segment, axis=0).astype(np.uint16)
        segment = segment.tolist()
        color = (0, 0, 0)
        for j, point in enumerate(segment):
            x1, y1, = point
            if j < len(segment) - 1:
                x2, y2 = segment[j + 1]
            else:
                x2, y2 = segment[0]

            cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=1)

        cv2.putText(img, "{}".format(unified_id), (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), thickness=1)

    cv2.imshow("{}".format("".join([args.parking_ground, "_heatmap"])), img)
    cv2.waitKey(0)
    if save_images:
        file_name = "".join([os.path.basename(file_path).split(".")[0], "_heatmap_result.jpg"])
        cv2.imwrite(os.path.join(args.image_save_dir, file_name), img)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    root_dir = os.path.abspath(".")
    sys.path.append(root_dir)

    args = get_args()

    json_label = read_label_file(label_file_path=args.label_file_path)

    parse_json_label(args=args, json_label=json_label)