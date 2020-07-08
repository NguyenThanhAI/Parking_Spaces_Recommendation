import sys
import os
import argparse
import shutil
from datetime import datetime
from tqdm import tqdm

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

from release_serving.enumerate_videos import enumerate_videos, get_time_from_videos


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_dir", type=str, required=True, help="Source video directory")
    parser.add_argument("--save_dir", type=str, required=True, help="Save video directory")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    videos_list = enumerate_videos(args.source_dir)

    day_dict = dict(map(lambda x: (x, get_time_from_videos(x)), videos_list))

    videos_filter = list(filter(lambda x: (day_dict[x] >= datetime(year=2019, month=9, day=26, hour=12) and day_dict[x] < datetime(year=2019, month=9, day=27, hour=12)) \
                                          or (day_dict[x] >= datetime(year=2019, month=9, day=29, hour=0) and day_dict[x] < datetime(year=2019, month=9, day=30, hour=0)) \
                                          or (day_dict[x] >= datetime(year=2019, month=10, day=27, hour=0) and day_dict[x] < datetime(year=2019, month=10, day=28, hour=0)) \
                                          or (day_dict[x] >= datetime(year=2019, month=10, day=31, hour=12) and day_dict[x] < datetime(year=2019, month=11, day=1, hour=12)) \
                                          or (day_dict[x] >= datetime(year=2019, month=11, day=24, hour=0) and day_dict[x] < datetime(year=2019, month=11, day=25, hour=0)) \
                                          or (day_dict[x] >= datetime(year=2019, month=11, day=28, hour=12) and day_dict[x] < datetime(year=2019, month=11, day=29, hour=12)), day_dict.keys()))

    for video in tqdm(videos_filter):
        save_directory = os.path.join(args.save_dir, *os.path.dirname(video).split(os.sep)[-3:])
        if not os.path.exists(save_directory):
            os.makedirs(save_directory, exist_ok=True)
        video_name = os.path.basename(video)
        shutil.copy(video, os.path.join(save_directory, video_name))
