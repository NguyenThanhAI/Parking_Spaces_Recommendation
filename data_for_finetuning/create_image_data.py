import os
import re
import argparse
from tqdm import tqdm
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--videos_dir", type=str, default=r"H:\04_前沢SA_上", help="Directory to videos dataset")
    parser.add_argument("--num_frames_per_video", type=int, default=1, help="Number of random frames sampled from video")
    parser.add_argument("--save_dir", type=str, default=r"H:\finetuning_dataset", help="Directory to created dataset")

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

    for video in tqdm(videos_list):
        #print(video)
        relative_dir_compose = os.path.dirname(video).split(os.sep)[len(dirname_compose):]
        video_name = os.path.basename(video).split(".")[0]
        relative_dir_compose.append(video_name)
        try:
            cap = cv2.VideoCapture(video)

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            #frame_index = np.random.choice(np.arange(length), size=args.num_frames_per_video, replace=False)
            frame_index = [int(0.1 * length), int(0.9 * length)]

            for index in frame_index:
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                _, img = cap.read()

                image_name = "_".join(relative_dir_compose) + "_" + str(index) + ".jpg"
                image_name = pattern.sub("", image_name)
                #print(image_name)
                cv2.imwrite(os.path.join(args.save_dir, image_name), img)

        except:
            print(video)


if __name__ == '__main__':
    main()
