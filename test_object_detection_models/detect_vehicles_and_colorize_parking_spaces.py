import os
import argparse

from itertools import groupby
from operator import itemgetter

import time
import json
from pathlib import Path
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN

from skimage.draw import polygon
from parking_spaces_assignment.utils import find_unique_values_and_frequency

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 4
    DETECTION_MIN_CONFIDENCE = 0.5


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, default=r"C:\Users\Thanh\Downloads")
    parser.add_argument("--image_path", type=str, default=r"C:\Users\Thanh\Downloads\mehdi-sakout-YcjIiFBXO1g-unsplash.jpg")
    parser.add_argument("--parking_spaces_config_json", type=str, default=r"C:\Users\Thanh\Downloads\202011\annotations\instances_default.json")
    parser.add_argument("--ios_threshold", type=float, default=0.5)
    parser.add_argument("--result_dir", type=str, default=r"results")

    args = parser.parse_args()

    return args


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


def json_to_parking_spaces_mask(args, json_label):
    assert isinstance(json_label, dict)
    images = json_label["images"]
    annotations = json_label["annotations"]
    assert isinstance(images, list) and isinstance(annotations, list)

    for image_id, items in groupby(images, key=itemgetter("id")):

        for item in items:
            print("Path of image id {0} is {1}".format(image_id, item["file_name"]))
            file_name = item["file_name"]
            width = item["width"]
            height = item["height"]
            file_path = os.path.join(args.dataset_dir, file_name)

        ps_mask_view = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8)

        parking_spaces = list(filter(lambda x: x["image_id"] == image_id, annotations))

        img_mask = -1 * np.ones(shape=[height, width])
        square_mask = {}
        pos_mask = {}

        for i, parking_space in enumerate(parking_spaces):
            segmentation = parking_space["segmentation"]
            id = parking_space["id"]
            segmentation = np.array(segmentation, dtype=np.uint16).reshape(-1, 2)
            cc, rr = segmentation.T
            rr, cc = polygon(rr, cc)
            img_mask[rr, cc] = id
            ps_mask_view[rr, cc] = np.random.randint(0, 255, size=[3])
            square_mask[id] = rr.shape[0]
            pos_mask[id] = (rr, cc)

        #cv2.imshow("Parking space mask", ps_mask_view)
        #cv2.waitKey(0)

    return img_mask, square_mask, pos_mask


if __name__ == '__main__':
    args = get_args()

    ROOT_DIR = Path(".")

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_cars_and_vehicles_0150.h5")

    RESULT_DIR = os.path.join(ROOT_DIR, "results")

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR, exist_ok=True)

    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

    model.load_weights(COCO_MODEL_PATH, by_name=True)

    img = cv2.imread(args.image_path)
    height, width = img.shape[:2]
    rgb_img = img[:, :, ::-1]

    start = time.time()
    results = model.detect([rgb_img], verbose=0)
    end = time.time()

    result = results[0]

    rois, scores, class_ids, masks = result["rois"], result["scores"], result["class_ids"], result["masks"]

    masks = masks.transpose(2, 0, 1)

    color = np.array([255, 255, 0], dtype=np.uint8)

    color_mask = 255 * np.ones_like(img, dtype=np.uint8)

    vehicle_mask = -1 * np.ones(shape=[height, width], dtype=np.uint8)
    vehicle_square_of_mask = {}
    vehicle_bbox = {}

    veh_id = 0
    for roi, mask in zip(rois, masks):
        vehicle_mask = np.where(mask, veh_id, vehicle_mask)
        vehicle_square_of_mask[veh_id] = np.count_nonzero(mask)
        y_min, x_min, y_max, x_max = roi
        vehicle_bbox[veh_id] = [x_min, y_min, x_max, y_max]
        veh_id += 1
        #cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    #cv2.imshow("Vehicle", img)
    #cv2.waitKey(0)

    #print(vehicle_square_of_mask, np.where(vehicle_mask >= 0))
    json_label = read_label_file(args.parking_spaces_config_json)
    parking_space_mask, parking_space_square_of_mask, pos_mask = json_to_parking_spaces_mask(args, json_label)
    #print(parking_space_square_of_mask)

    num_vehicles = masks.shape[0]

    unified_id_to_vehicle_id_ios = {}
    vehicle_id_to_unified_id_ios = {}

    for veh_id in vehicle_bbox:
        x_min, y_min, x_max, y_max = vehicle_bbox[veh_id]
        cropped_ps_mask = parking_space_mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_vh_mask = vehicle_mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)

        inter_dict = find_unique_values_and_frequency(cropped_mask, veh_id, False)
        #print(inter_dict)
        for ps_veh in inter_dict:
            uid, vid = ps_veh
            uid = int(uid)
            vid = int(vid)
            inter = inter_dict[ps_veh]
            ios = inter / parking_space_square_of_mask[uid]

            if uid not in unified_id_to_vehicle_id_ios:
                unified_id_to_vehicle_id_ios[uid] = {}
            if vid not in unified_id_to_vehicle_id_ios[uid]:
                unified_id_to_vehicle_id_ios[uid][vid] = inter / parking_space_square_of_mask[uid]
            else:
                assert unified_id_to_vehicle_id_ios[uid][vid] == (
                            inter / parking_space_square_of_mask[uid]), "ios of 2 times is not equal"
            if vid not in vehicle_id_to_unified_id_ios:
                vehicle_id_to_unified_id_ios[vid] = {}
            if uid not in vehicle_id_to_unified_id_ios[vid]:
                vehicle_id_to_unified_id_ios[vid][uid] = inter / parking_space_square_of_mask[uid]
            else:
                assert vehicle_id_to_unified_id_ios[vid][uid] == (
                            inter / parking_space_square_of_mask[uid]), "ios of 2 times is not equal"

    #print(unified_id_to_vehicle_id_ios, vehicle_id_to_unified_id_ios)

    has_vehicle = []

    for veh_id in vehicle_id_to_unified_id_ios:
        if len(vehicle_id_to_unified_id_ios[veh_id]) > 0:
            for uid in vehicle_id_to_unified_id_ios[veh_id]:
                if vehicle_id_to_unified_id_ios[veh_id][uid] > args.ios_threshold:
                    if uid not in has_vehicle:
                        has_vehicle.append(uid)
                        rr, cc = pos_mask[uid]
                        color_mask[rr, cc] = (0, 0, 255)

    for uid in parking_space_square_of_mask:
        if uid not in has_vehicle:
            rr, cc = pos_mask[uid]
            color_mask[rr, cc] = (0, 255, 0)

    #cv2.imshow("Anh", color_mask)
    #cv2.waitKey(0)

    color_mask = np.where(np.expand_dims(vehicle_mask, axis=-1) >= 0, np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
    #print(color_mask)
    #cv2.imshow("Color mask", color_mask)
    #cv2.waitKey(0)

    results = cv2.addWeighted(img, 0.6, color_mask, 0.4, 0)
    #cv2.imshow("Results", results)
    #cv2.waitKey(0)

    cv2.imwrite(os.path.join(RESULT_DIR, os.path.basename(args.image_path)), results)
