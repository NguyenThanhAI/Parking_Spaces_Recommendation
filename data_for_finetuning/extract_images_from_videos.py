import os
import argparse
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video_path", type=str, default=r"C:\Users\Thanh_Tuyet\Downloads\2019-10-27_12-00-00.mp4", help="Path to video")
    parser.add_argument("--save_dir", type=str, default=r"C:\Users\Thanh_Tuyet\Downloads\Images", help="Image saving directory")
    parser.add_argument("--frame_step", type=int, default=1000, help="Frame step")

    args = parser.parse_args()

    return args


def extract_images(args):
    video_path = args.video_path

    cap = cv2.VideoCapture(video_path)

    save_dir = os.path.join(args.save_dir, os.path.basename(video_path).split(".")[0])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    count = 0

    while True:
        ret, image = cap.read()
        print(count)

        if not ret:
            break

        if count % args.frame_step == 0:
            cv2.imwrite(os.path.join(save_dir, "img_%06d.jpg" % count), image)

        count += 1

if __name__ == '__main__':
    args = get_args()
    extract_images(args)