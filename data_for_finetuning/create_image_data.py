import os
import re
from itertools import groupby, chain
import argparse
from tqdm import tqdm
import numpy as np
import cv2

np.random.seed(1000)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--videos_dir", type=str, default=r"G:\04_前沢SA_上", help="Directory to videos dataset")
    parser.add_argument("--num_frames_per_video", type=int, default=1, help="Number of random frames sampled from video")
    parser.add_argument("--number_chosen_hours_per_day", type=int, default=12, help="Number of hours chosen per day")
    parser.add_argument("--save_dir", type=str, default=r"F:\finetuning_dataset\SA", help="Directory to created dataset")

    args = parser.parse_args()

    return args


def enumerate_videos_from_dir(args):
    videos_list = []
    for dirs, _, files in os.walk(args.videos_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mkv")):
                videos_list.append(os.path.join(dirs, file))

    return videos_list


def main():

    args = get_args()

    pattern = re.compile(u"[カメラ]", re.UNICODE)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    dirname_compose = args.videos_dir.split(os.sep)

    videos_list = enumerate_videos_from_dir(args)
    for day, videos in groupby(videos_list, key=lambda x: os.path.join(*os.path.dirname(x).split(os.sep)[:-1])):
        # Chỉ sử dụng videos được đúng một lần????????????????????????????
        hours_dict = {}
        for video in videos:
            if os.path.basename(video).split(".")[0] not in hours_dict:
                hours_dict[os.path.basename(video).split(".")[0]] = []
            hours_dict[os.path.basename(video).split(".")[0]].append(video)
        chosen_hours = np.random.choice(list(hours_dict.keys()), size=args.number_chosen_hours_per_day, replace=False)
        chosen_videos = list(chain.from_iterable(list(map(lambda x: hours_dict[x], chosen_hours))))
        for vid in tqdm(chosen_videos):
            relative_dir_compose = os.path.dirname(vid).split(os.sep)[len(dirname_compose):]
            video_name = os.path.basename(vid).split(".")[0]
            relative_dir_compose.append(video_name)
            try:
                cap = cv2.VideoCapture(vid)

                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_index = np.random.choice(np.arange(length), size=args.num_frames_per_video, replace=False)

                for index in frame_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    _, img = cap.read()

                    image_name = "_".join(relative_dir_compose) + "_" + str(index) + ".jpg"
                    image_name = pattern.sub("", image_name)
                    # print(image_name)
                    cv2.imwrite(os.path.join(args.save_dir, image_name), img)

            except:
                print(vid)


if __name__ == '__main__':
    main()
