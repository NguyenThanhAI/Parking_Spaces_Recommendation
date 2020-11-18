import os
import argparse

from itertools import groupby
from operator import itemgetter
from tqdm import tqdm

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
from vehicle_tracking.utils import non_maximum_suppression

from scipy.interpolate import NearestNDInterpolator
from sklearn.neighbors import KDTree


class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.0


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, default=r"C:\Users\Thanh\Downloads\production_ID_3858833.mp4")
    parser.add_argument("--parking_spaces_config_json", type=str, default=r"C:\Users\Thanh\Downloads\task_parking_11_19-2020_11_17_03_18_12-coco_1.0\annotations\instances_default.json")
    parser.add_argument("--ios_threshold", type=float, default=0.3)
    parser.add_argument("--result_dir", type=str, default=r"results")
    parser.add_argument("--is_nms", type=str2bool, default=True, help="Run NMS or not")
    parser.add_argument("--is_filterboxes", type=str2bool, default=True, help="Filter boxes or not")
    parser.add_argument("--is_showframe", type=str2bool, default=False, help="Show result or not")

    args = parser.parse_args()

    return args


def filter_boxes(rois, scores, class_ids, masks, size):
    height, width, _ = size
    new_rois = []
    new_scores = []
    new_class_ids = []
    new_maskes = []
    for idx, mask in enumerate(masks):
        rr, cc = np.where(mask)
        y_min, y_max = np.min(rr), np.max(rr)
        x_min, x_max = np.min(cc), np.max(cc)
        if x_max - x_min > 0.07 * width or y_max - y_min > 0.12 * height or x_max - x_min < 0.04 * width or y_max - y_min < 0.04 * height or (y_max - y_min) * (x_max - x_min) < 0.003 * height * width:
            continue
        new_rois.append(rois[idx])
        new_scores.append(scores[idx])
        new_class_ids.append(class_ids[idx])
        new_maskes.append(masks[idx])

    return np.array(new_rois), np.array(new_scores), np.array(new_class_ids), np.array(new_maskes)


def read_label_file(label_file_path):
    with open(label_file_path, "r") as f:
        json_label = json.load(f)

    return json_label


def json_to_parking_spaces_mask(args, json_label):
    assert isinstance(json_label, dict)
    images = json_label["images"]
    annotations = json_label["annotations"]
    assert isinstance(images, list) and isinstance(annotations, list)

    image_to_mask = {}

    for image_id, items in groupby(images, key=itemgetter("id")):

        for item in items:
            print("Path of image id {0} is {1}".format(image_id, item["file_name"]))
            file_name = item["file_name"]
            frame_id = int(file_name.split(".")[0].split("_")[1])
            image_to_mask[frame_id] = {}
            width = item["width"]
            height = item["height"]

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

        image_to_mask[frame_id]["img_mask"] = img_mask
        image_to_mask[frame_id]["square_mask"] = square_mask
        image_to_mask[frame_id]["pos_mask"] = pos_mask

        #cv2.imshow(str(frame_id), ps_mask_view)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    return image_to_mask


if __name__ == '__main__':
    args = get_args()

    ROOT_DIR = Path(".")

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_cars_and_vehicles_0008.h5")

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir, exist_ok=True)

    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())

    model.load_weights(COCO_MODEL_PATH, by_name=True)

    json_label = read_label_file(args.parking_spaces_config_json)
    image_to_mask = json_to_parking_spaces_mask(args, json_label)

    frame_milestone = np.array(list(image_to_mask.keys()))

    tree = KDTree(np.expand_dims(frame_milestone, axis=1))

    #index = tree.query([[321]], return_distance=False)
#
    #respective_milestone = frame_milestone[np.squeeze(index)]
#
    #print(respective_milestone)

    cap = cv2.VideoCapture(args.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(os.path.join(args.result_dir, "result.mp4"), fourcc, fps, (width, height))

    #frame_id = 0
    for frame_id in tqdm(range(num_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        rgb_img = frame[:, :, ::-1]

        start = time.time()
        results = model.detect([rgb_img], verbose=0)
        end = time.time()

        result = results[0]

        rois, scores, class_ids, masks = result["rois"], result["scores"], result["class_ids"], result["masks"]

        masks = masks.transpose(2, 0, 1)

        if args.is_nms:
            chosen_indices = non_maximum_suppression(bboxes=rois, scores=scores, max_bbox_overlap=0.3)
            rois = rois[chosen_indices]
            scores = scores[chosen_indices]
            class_ids = class_ids[chosen_indices]
            masks = masks[chosen_indices]

        if args.is_filterboxes:
            rois, scores, class_ids, masks = filter_boxes(rois=rois, scores=scores, class_ids=class_ids, masks=masks, size=frame.shape)

        color = np.array([255, 255, 0], dtype=np.uint8)

        color_mask = 255 * np.ones_like(frame, dtype=np.uint8)

        vehicle_mask = -1 * np.ones(shape=[height, width], dtype=np.uint8)
        vehicle_square_of_mask = {}
        vehicle_bbox = {}

        veh_id = 0
        for roi, mask in zip(rois, masks):
            vehicle_mask = np.where(mask, veh_id, vehicle_mask)
            vehicle_square_of_mask[veh_id] = np.count_nonzero(mask)
            #y_min, x_min, y_max, x_max = roi
            rr, cc = np.where(mask)
            y_min, y_max = np.min(rr), np.max(rr)
            x_min, x_max = np.min(cc), np.max(cc)
            vehicle_bbox[veh_id] = [x_min, y_min, x_max, y_max]
            veh_id += 1

        index = tree.query([[frame_id]], return_distance=False)

        respective_milestone = frame_milestone[np.squeeze(index)]
        print("Frame id: {}, Respective milestone: {}".format(frame_id, respective_milestone))
        parking_space_mask, parking_space_square_of_mask, pos_mask = image_to_mask[respective_milestone]["img_mask"], image_to_mask[respective_milestone]["square_mask"], image_to_mask[respective_milestone]["pos_mask"]
        # print(parking_space_square_of_mask)

        num_vehicles = masks.shape[0]

        unified_id_to_vehicle_id_ios = {}
        vehicle_id_to_unified_id_ios = {}

        for veh_id in vehicle_bbox:
            x_min, y_min, x_max, y_max = vehicle_bbox[veh_id]
            cropped_ps_mask = parking_space_mask[y_min:y_max + 1, x_min:x_max + 1]
            cropped_vh_mask = vehicle_mask[y_min:y_max + 1, x_min:x_max + 1]
            cropped_mask = np.stack((cropped_ps_mask, cropped_vh_mask), axis=2)

            inter_dict = find_unique_values_and_frequency(cropped_mask, veh_id, False)
            # print(inter_dict)
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

        # print(unified_id_to_vehicle_id_ios, vehicle_id_to_unified_id_ios)

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

        # cv2.imshow("Anh", color_mask)
        # cv2.waitKey(0)

        #color_mask = np.where(np.expand_dims(vehicle_mask, axis=-1) >= 0,
        #                      np.array([255, 0, 0], dtype=np.uint8)[np.newaxis, np.newaxis, :], color_mask)
        color_mask = np.where(np.expand_dims(vehicle_mask, axis=-1) >= 0,
                              0, color_mask)
        # print(color_mask)
        # cv2.imshow("Color mask", color_mask)
        # cv2.waitKey(0)

        results = np.where(color_mask > 0, cv2.addWeighted(frame, 0.5, color_mask, 0.5, 0), frame)

        for veh_id in vehicle_bbox:
        #for roi in rois:
            x_min, y_min, x_max, y_max = vehicle_bbox[veh_id]
            #y_min, x_min, y_max, x_max = roi
            #roi_width = round((x_max - x_min) / width, 3)
            #roi_height = round((y_max - y_min) / height, 3)
            #square = round(roi_width * roi_height, 3)
            #cv2.putText(results, "roi_width: {}, roi_height: {}, square: {}".format(roi_width, roi_height, square), org=(x_min + 50, y_min + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3, color=(0, 255, 0))
            cv2.rectangle(results, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

        if args.is_showframe:
            cv2.imshow("Results", results)
            cv2.waitKey(0)

        writer.write(results)

        #frame_id += 1

    writer.release()
