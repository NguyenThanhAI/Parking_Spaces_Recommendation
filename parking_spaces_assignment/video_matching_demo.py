import sys
import os

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

import argparse
from parking_spaces_assignment.matcher import Matcher


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_source", type=str, default=r"G:\04_前沢SA_上\201911\20191130\カメラ1\2019-11-30_10-00-00.mp4", help="Path to video")
    parser.add_argument("--video_output_dir", type=str, default=r"F:\\", help="Path to output video")
    parser.add_argument("--model_arch", type=str, choices=["mask_rcnn", "yolact"], default="mask_rcnn", help="Model detector")
    parser.add_argument("--checkpoint_name", type=str, default="mask_rcnn_cars_and_vehicles_0008.h5", help="Checkpoint name")
    parser.add_argument("--cuda", type=bool, default=False, help="Use cuda or not")
    parser.add_argument("--ios_threshold", type=float, default=0.1, help="Threshold of ios to considered possibility of matching")
    parser.add_argument("--iov_threshold", type=float, default=0.1, help="Threshold of iov to considered posibility of matching")
    parser.add_argument("--is_showframe", type=bool, default=True, help="Show result or not")
    parser.add_argument("--parking_ground", type=str, default="parking_ground_SA", help="Parking ground")
    parser.add_argument("--cam_list", type=str, default="cam_1", help="Cam")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    matcher = Matcher(active_cams=[args.cam_list], parking_ground=args.parking_ground, model_arch=args.model_arch, checkpoint_name=args.checkpoint_name)
    matcher.video_match(video_source=args.video_source, is_savevideo=True, save_dir=args.video_output_dir,
                        cam=args.cam_list, ios_threshold=args.ios_threshold, iov_threshold=args.iov_threshold, is_tracking=True, is_showframe=args.is_showframe)
