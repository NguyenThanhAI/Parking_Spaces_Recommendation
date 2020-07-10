import sys
import os
from datetime import datetime

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

import argparse
from parking_spaces_assignment.matcher import Matcher
from release_serving.enumerate_videos import get_sequence_of_video_list


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

    parser.add_argument("--video_dir", type=str, default=r"D:\SA", help="Path to video")
    parser.add_argument("--video_output_dir", type=str, default=r"E:\\", help="Path to output video")
    parser.add_argument("--database_dir", type=str, default=r"F:\\", help="Database directory")
    parser.add_argument("--model_arch", type=str, choices=["mask_rcnn", "yolact"], default="mask_rcnn", help="Model detector")
    parser.add_argument("--checkpoint_name", type=str, default="mask_rcnn_cars_and_vehicles_0008.h5", help="Checkpoint name")
    parser.add_argument("--detection_vehicle_thresh", type=float, default=0.4, help="Detection threshold")
    parser.add_argument("--cuda", type=bool, default=False, help="Use cuda or not")
    parser.add_argument("--ios_threshold", type=float, default=0.1, help="Threshold of ios to considered possibility of matching")
    parser.add_argument("--iov_threshold", type=float, default=0.1, help="Threshold of iov to considered posibility of matching")
    parser.add_argument("--is_showframe", type=str2bool, nargs="?", const=True, default=True, help="Show result or not")
    parser.add_argument("--parking_ground", type=str, default="parking_ground_SA", help="Parking ground")
    parser.add_argument("--cam_list", type=str, default="cam_1,cam_2,cam_3", help="Cam")
    parser.add_argument("--run_multiprocessing", type=str2bool, nargs="?", const=True, default=True, help="Run multiprocessing or not")
    parser.add_argument("--use_config_considered_area", type=str2bool, nargs="?", const=True, default=False, help="Use config areas or not")
    parser.add_argument("--use_mysql", type=str2bool, nargs="?", const=True, default=True, help="Use config areas or not")
    parser.add_argument("--host", type=str, default="18.181.144.207", help="Host ip")
    parser.add_argument("--port", type=str, default="3306", help="Port")
    parser.add_argument("--user", type=str, default="edge_matrix", help="User")
    parser.add_argument("--passwd", type=str, default="edgematrix", help="Password")
    parser.add_argument("--database_file", type=str, default="edge_matrix_thanh", help="Database file")
    parser.add_argument("--reset_table", type=str2bool, default=False, help="Reset tables or not")

    args = parser.parse_args()

    return args


def run_function(args, time_intervals):
    cam_list = [cam for cam in args.cam_list.split(",")]
    for i, interval in enumerate(time_intervals):
        sequence_video_source_list = get_sequence_of_video_list(args.video_dir, interval)
        if i == 0 and args.reset_table:
            reset_table = True
        else:
            reset_table = False
        print("Interval {}, reset table {}, time: {}".format(i, reset_table, datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        matcher = Matcher(active_cams=cam_list, parking_ground=args.parking_ground, model_arch=args.model_arch,
                          checkpoint_name=args.checkpoint_name, detection_vehicle_thresh=args.detection_vehicle_thresh,
                          run_multiprocessing=args.run_multiprocessing,
                          use_config_considered_area=args.use_config_considered_area)
        matcher.sequence_video_match(sequence_video_source_list=sequence_video_source_list, is_savevideo=True, save_dir=args.video_output_dir,
                                     database_dir=args.database_dir,
                                     cam_list=cam_list, ios_threshold=args.ios_threshold, iov_threshold=args.iov_threshold,
                                     is_tracking=True, is_showframe=args.is_showframe,
                                     use_mysql=args.use_mysql, host=args.host, port=args.port, user=args.user,
                                     passwd=args.passwd, database_file=args.database_file, reset_table=reset_table)


if __name__ == '__main__':
    time_intervals = [(datetime(year=2019, month=9, day=26, hour=12), datetime(year=2019, month=9, day=27, hour=12)),
                      (datetime(year=2019, month=9, day=29, hour=0), datetime(year=2019, month=9, day=30, hour=0)),
                      (datetime(year=2019, month=10, day=27, hour=0), datetime(year=2019, month=10, day=28, hour=0)),
                      (datetime(year=2019, month=10, day=31, hour=12), datetime(year=2019, month=11, day=1, hour=12)),
                      (datetime(year=2019, month=11, day=24, hour=0), datetime(year=2019, month=11, day=25, hour=0)),
                      (datetime(year=2019, month=11, day=28, hour=12), datetime(year=2019, month=11, day=29, hour=12))]
    args = get_args()

    run_function(args, time_intervals)
