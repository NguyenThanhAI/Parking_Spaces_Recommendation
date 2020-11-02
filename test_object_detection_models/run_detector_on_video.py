import sys
import os

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

import argparse
import numpy as np
import cv2

import torch

from load_videos.videostream import QueuedStream
from vehicle_tracking.multiprocess_vehicle_detector import MultiProcessVehicleDetector
from vehicle_tracking.vehicle_detector_1 import VehicleDetector
from vehicle_tracking.utils import non_maximum_suppression, iou_mat


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

    parser.add_argument("--video_source", type=str, help="Path to video source")
    parser.add_argument("--model_arch", type=str, choices=["mask_rcnn", "yolact"], default="mask_rcnn", help="Model detector")
    parser.add_argument("--checkpoint_name", type=str, default="mask_rcnn_cars_and_vehicles_0008.h5", help="Checkpoint name")
    parser.add_argument("--detection_vehicle_thresh", type=float, default=0.4, help="Detection threshold")
    parser.add_argument("--is_savevideo", type=str2bool, default=True, help="Show result or not")
    parser.add_argument("--save_dir", type=str, help="Show result or not")
    parser.add_argument("--run_multiprocessing", type=str2bool, default=False, help="Run multiprocessing or not")
    parser.add_argument("--is_nms", type=str2bool, default=False, help="Run NMS or not")
    parser.add_argument("--is_filterboxes", type=str2bool, default=False, help="Filter boxes or not")
    parser.add_argument("--is_showframe", type=str2bool, default=True, help="Show result or not")

    args = parser.parse_args()

    return args


def filter_boxes(rois, scores, class_ids, masks, size):
    height, width, _ = size
    new_rois = []
    new_scores = []
    new_class_ids = []
    new_maskes = []
    for idx, roi in enumerate(rois):
        y_min, x_min, y_max, x_max = roi
        if x_max - x_min > 0.4 * width or y_max - y_min > 0.7 * height or x_max - x_min < 0.05 * width or y_max - y_min < 0.05 * height or (y_max - y_min) * (x_max - x_min) < 0.01 * height * width:
            continue
        new_rois.append(rois[idx])
        new_scores.append(scores[idx])
        new_class_ids.append(class_ids[idx])
        new_maskes.append(masks[idx])

    return np.array(new_rois), np.array(new_scores), np.array(new_class_ids), np.array(new_maskes)


def filter_iou(rois, scores, class_ids, masks):
    hold_idx = list(range(len(rois)))
    iou = iou_mat(rois, rois)
    iu = np.tril_indices(iou.shape[0])
    iou[iu] = 0.
    pairs_idx = np.argwhere(iou)
    for pair in pairs_idx:
        a_idx, b_idx = pair
        a_roi = rois[a_idx]
        b_roi = rois[b_idx]
        a_y_min, a_x_min, a_y_max, a_x_max = a_roi
        b_y_min, b_x_min, b_y_max, b_x_max = b_roi
        a_area = (a_y_max - a_y_min + 1) * (a_x_max - a_x_min + 1)
        b_area = (b_y_max - b_y_min + 1) * (b_x_max - b_x_min + 1)
        if a_area >= b_area:
            large_idx = a_idx
            tiny_idx = b_idx
        else:
            large_idx = b_idx
            tiny_idx = a_idx

        if iou[a_idx, b_idx] > 0.05 and rois[large_idx][0] < rois[tiny_idx][0] and rois[large_idx][1] < rois[tiny_idx][1] and rois[large_idx][2] < rois[tiny_idx][2]:
            hold_idx.remove(tiny_idx)

    return hold_idx


if __name__ == '__main__':
    args = get_args()

    class_id_to_name = {1: "Car", 2: "Truck", 3: "Bus", 4: "Bicycle"}

    if args.run_multiprocessing:
        print("Run mutiprocessing")
        detector = MultiProcessVehicleDetector(checkpoint_name=args.checkpoint_name, detection_vehicle_thresh=args.detection_vehicle_thresh, model_arch=args.model_arch)
        detector.start()
        detector.warm_up()
    else:
        print("Don't run multiprocessing")
        detector = VehicleDetector(checkpoint_name=args.checkpoint_name,
                                   detection_vehicle_thresh=args.detection_vehicle_thresh,
                                   model_arch=args.model_arch,
                                   cuda=torch.cuda.is_available())
    run = False

    cap = cv2.VideoCapture(args.video_source)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.is_savevideo:
        assert args.save_dir, "When save video, save_dir cannot be None"
        fps = 5
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        if args.video_source.endswith((".mp4", ".avi")):
            video_name = os.path.basename(args.video_source)
        else:
            video_name = "save_webcam.mp4"

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir, exist_ok=True)

        output = cv2.VideoWriter(os.path.join(args.save_dir, video_name), fourcc, fps, (width, height))

    cap.release()

    stream = QueuedStream(args.video_source, "cam_1")

    stream.start()

    if not stream.isOpened():
        print("Can not open video: {}".format(args.video_source))
        detector.stop()
        raise StopIteration

    run = True

    while run:
        ret, frame, frame_id, time_stamp, cam_stream = stream.read()

        if not ret:
            run = False
            break

        if args.run_multiprocessing:
            detector.put_frame(frame_id, frame, time_stamp, cam_stream)

            rois, scores, class_ids, masks, frame_id, time_stamp, cam_detect = detector.get_result()
            print(rois.shape, scores.shape, class_ids.shape, masks.shape)
        else:
            rois, scores, class_ids, masks = detector(frame, parking_ground=None, cam=None)

        if args.is_nms:
            chosen_indices = non_maximum_suppression(bboxes=rois, scores=scores, max_bbox_overlap=0.3)
            rois = rois[chosen_indices]
            scores = scores[chosen_indices]
            class_ids = class_ids[chosen_indices]
            masks = masks[chosen_indices]

        if args.is_filterboxes:
            rois, scores, class_ids, masks = filter_boxes(rois=rois, scores=scores, class_ids=class_ids, masks=masks, size=frame.shape)

            hold_idx = filter_iou(rois, scores, class_ids, masks)
            rois = rois[hold_idx]
            scores = scores[hold_idx]
            class_ids = class_ids[hold_idx]
            masks = masks[hold_idx]

        if args.model_arch == "mask_rcnn":
            class_id_list = [1, 2, 3, 4]
        else:
            class_id_list = [2, 5, 7]

        color = np.array([255, 0, 0], dtype=np.uint8)

        color_mask = np.zeros_like(frame, dtype=np.uint8)

        for det_id, (roi, score, class_id, mask) in enumerate(zip(rois, scores, class_ids, masks)):
            if score >= args.detection_vehicle_thresh and class_id in class_id_list:
                #y_min, x_min, y_max, x_max = roi
                #hr = (y_max - y_min) / height
                #wr = (x_max - x_min)/ width
                #sr = (y_max - y_min) * (x_max - x_min) / (width * height)
                rr, cc = np.where(mask)
                y_min, y_max = np.min(rr), np.max(rr)
                x_min, x_max = np.min(cc), np.max(cc)
                color_mask = np.where(mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                #cv2.putText(frame, str(round(score, 3)) + " " + str(round(hr, 3)) + " " + str(round(wr, 3)) + " " + str(round(sr, 3)), org=(x_min - 80, y_min + 20), fontScale=0.35, color=(0, 0, 255), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                cv2.putText(frame, str(class_id_to_name[class_id]) + " " + str(round(score, 3)), org=(x_min + 20, y_min + 20), fontScale=0.35, color=(0, 0, 255), thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

        frame = np.where(color_mask > 0, cv2.addWeighted(frame, 0.3, color_mask, 0.7, 0), frame)

        if args.is_savevideo:
            output.write(frame)
        if args.is_showframe:
            cv2.imshow("Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            run = False
            if args.run_multiprocessing:
                detector.stop()
            stream.stop()
            cv2.destroyAllWindows()
            break

    if args.is_savevideo:
        output.release()
        print("Save video")

    print("Done")
