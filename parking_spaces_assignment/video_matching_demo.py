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
    parser.add_argument("--threshold", type=float, default=0.3, help="Threshold of ios to considered possibility of matching")
    parser.add_argument("--is_showframe", type=bool, default=True, help="Show result or not")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    matcher = Matcher(active_cams=["cam_1"])
    matcher.video_match(video_source=args.video_source, is_savevideo=True, save_dir=args.video_output_dir,
                        cam="cam_1", threshold=args.threshold, is_tracking=True, is_showframe=args.is_showframe)
