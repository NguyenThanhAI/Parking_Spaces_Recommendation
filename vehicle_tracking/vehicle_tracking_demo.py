import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from vehicle_tracking.vehicle_detector import VehicleDetector
from vehicle_tracking.vehicle_tracker import VehicleTracker

ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_source", type=str, default=r"G:\04_前沢SA_上\201911\20191130\カメラ1\2019-11-30_10-00-00.mp4", help="Path to demo video")
    parser.add_argument("--video_output_dir", type=str, default=r"F:\\", help="Path to output video")
    parser.add_argument("--is_showframe", type=str, default=True, help="Show result or not")
    parser.add_argument("--detection_vehicle_thresh", type=float, default=0.2)
    parser.add_argument("--inactive_steps_before_removed", type=int, default=10)
    parser.add_argument("--reid_iou_threshold", type=float, default=0.3)
    parser.add_argument("--max_traject_steps", type=int, default=50)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    detector = VehicleDetector()
    tracker = VehicleTracker(detection_vehicle_thresh=args.detection_vehicle_thresh,
                             inactive_steps_before_removed=args.inactive_steps_before_removed,
                             reid_iou_threshold=args.reid_iou_threshold,
                             max_traject_steps=args.max_traject_steps,
                             parking_ground="parking_ground_SA",
                             cam="cam_1")

    if not os.path.exists(args.video_output_dir):
        os.makedirs(args.video_output_dir)

    video_name = os.path.basename(args.video_source)

    cap = cv2.VideoCapture(args.video_source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output = cv2.VideoWriter(os.path.join(args.video_output_dir, video_name), fourcc, fps, (height, width))
    stopped = False
    for i in tqdm(range(length)):
        if not stopped:
            ret, frame = cap.read()

            if not ret:
                stopped = True
                continue

            vehicle_detections = detector(frame)
            tracker.step(vehicle_detections=vehicle_detections)
            vehicle_tracks = tracker.get_result()

            for vehicle_track in vehicle_tracks:
                bbox = vehicle_track.bbox
                track_id = vehicle_track.track_id
                class_id = vehicle_track.class_id
                x_min, y_min, x_max, y_max = bbox
                result = "Track id: {}, class id: {}".format(track_id, class_id)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 1)
                cv2.putText(frame, result, (x_min + int(0.05*(x_max - x_min)), y_min - int(0.05*(y_max - y_min))), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)

            #color = np.array([255, 255, 0], dtype=np.uint8)
            #color_mask = np.zeros_like(frame, dtype=np.uint8)
            #for vehicle_detection in vehicle_detections:
            #    bbox = vehicle_detection.bbox
            #    detection_id = vehicle_detection.detection_id
            #    class_id = vehicle_detection.class_id
            #    x_min, y_min, x_max, y_max = bbox
            #    mask = vehicle_detection.mask
            #    color_mask = np.where(mask[:, :, np.newaxis], color[np.newaxis, np.newaxis, :], color_mask)
            #    result = "Detection id: {}, class id: {}".format(detection_id, class_id)
            #    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            #    cv2.putText(frame, result, (x_min + int(0.4 * (x_max - x_min)), y_min - int(0.05 * (y_max - y_min))), cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 0), 1)
            #frame = np.where(color_mask > 0, cv2.addWeighted(color_mask, 0.5, frame, 0.5, 0), frame)
            output.write(frame)
            if args.is_showframe:
                cv2.imshow("", frame)
            if cv2.waitKey(1) == ord("q"):
                stopped = False
        else:
            output.release()
            cv2.destroyAllWindows()
            print("Save video and exit")
            break
