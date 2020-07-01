import sys
import os

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

import argparse
from parking_spaces_assignment.matcher import Matcher


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

    parser.add_argument("--video_source_list", type=str, default=r"G:\04_前沢SA_上\201911\20191130\カメラ1\2019-11-30_10-00-00.mp4", help="Path to video")
    parser.add_argument("--video_output_dir", type=str, default=r"F:\\", help="Path to output video")
    parser.add_argument("--model_arch", type=str, choices=["mask_rcnn", "yolact"], default="mask_rcnn", help="Model detector")
    parser.add_argument("--checkpoint_name", type=str, default="mask_rcnn_cars_and_vehicles_0008.h5", help="Checkpoint name")
    parser.add_argument("--detection_vehicle_thresh", type=float, default=0.4, help="Detection threshold")
    parser.add_argument("--cuda", type=bool, default=False, help="Use cuda or not")
    parser.add_argument("--ios_threshold", type=float, default=0.1, help="Threshold of ios to considered possibility of matching")
    parser.add_argument("--iov_threshold", type=float, default=0.1, help="Threshold of iov to considered posibility of matching")
    parser.add_argument("--is_showframe", type=str2bool, nargs="?", const=True, default=True, help="Show result or not")
    parser.add_argument("--parking_ground", type=str, default="parking_ground_SA", help="Parking ground")
    parser.add_argument("--cam_list", type=str, default="cam_1", help="Cam")
    parser.add_argument("--run_multiprocessing", type=str2bool, nargs="?", const=True, default=True, help="Run multiprocessing or not")
    parser.add_argument("--use_config_considered_area", type=str2bool, nargs="?", const=True, default=False, help="Use config areas or not")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    video_source_list = [video for video in args.video_source_list.split(",")]
    cam_list = [cam for cam in args.cam_list.split(",")]
    assert len(video_source_list) == len(cam_list)
    matcher = Matcher(active_cams=cam_list, parking_ground=args.parking_ground, model_arch=args.model_arch,
                      checkpoint_name=args.checkpoint_name, detection_vehicle_thresh=args.detection_vehicle_thresh,
                      run_multiprocessing=args.run_multiprocessing,
                      use_config_considered_area=args.use_config_considered_area)
    matcher.video_match(video_source_list=video_source_list, is_savevideo=True, save_dir=args.video_output_dir,
                        cam_list=cam_list, ios_threshold=args.ios_threshold, iov_threshold=args.iov_threshold,
                        is_tracking=True, is_showframe=args.is_showframe)
