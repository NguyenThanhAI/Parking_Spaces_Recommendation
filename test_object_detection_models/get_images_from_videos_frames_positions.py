import sys
import os
import re
import argparse
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, help="Path to video")
    parser.add_argument("--min_sec", type=str, help="Path to video")
    parser.add_argument("--save_dir", type=str, default="images", help="Path to save dir")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    root_dir = os.path.abspath(".")
    sys.path.append(root_dir)

    pattern = re.compile(u"[カメラ]", re.UNICODE)
    args = get_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = args.min_sec.split(",")
    frame_index = list(map(lambda x: int(x), frame_index))
    frame_index = int((frame_index[0] * 60 + frame_index[1]) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    ret, frame = cap.read()
    if not ret:
        print("Out of range frame")

    relative_dir_compose = os.path.dirname(args.video_path).split(os.sep)[2:]
    print(relative_dir_compose)
    video_name = os.path.basename(args.video_path).split(".")[0]
    relative_dir_compose.append(video_name)

    image_name = "_".join(relative_dir_compose) + "_" + str(frame_index) + ".jpg"
    image_name = pattern.sub("", image_name)

    cv2.imwrite(os.path.join(args.save_dir, image_name), frame)
