import os
from itertools import groupby
import argparse
import numpy as np
import cv2


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--videos_dir", type=str, default=r"G:", help="Directory of dataset video")
    parser.add_argument("--videos_path", type=str, help="Path to video")

    args = parser.parse_args()

    return args


def enumerate_videos(videos_dir):
    videos_list = []
    for dirs, _, files in os.walk(videos_dir):
        for file in files:
            if file.endswith(".mp4"):
                videos_list.append(os.path.join(dirs, file))

    return videos_list


def enumerate_template(videos_dir):
    videos_list = enumerate_videos(videos_dir)
    template_dict = {}
    for template, elements in groupby(videos_list, key=lambda x: os.path.join(*os.path.dirname(x).split(os.sep)[:-1], os.path.basename(x))):
        if template not in template_dict:
            template_dict[template] = []
        for element in elements:
            template_dict[template].append(element)
    return template_dict


#if __name__ == '__main__':
#    args = get_args()
#    template_dict = enumerate_template(args.videos_dir)
#    print("template_dict {}".format(template_dict))
#
#    assert all(map(lambda x: len(template_dict[x]) == 4, template_dict.keys())), "A video must belong to 4 cameras"
