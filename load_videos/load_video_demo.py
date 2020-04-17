import sys
import os
import argparse
import numpy as np
import cv2

from load_videos.enumerate_videos import get_args
from load_videos.videos_utils import noThreading, threadVideoGet, threadVideoShow, threadBoth

if __name__ == '__main__':
    args = get_args()

    noThreading(args.videos_path)