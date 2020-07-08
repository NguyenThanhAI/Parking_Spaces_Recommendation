import os
from itertools import groupby
from operator import itemgetter
from datetime import datetime


def enumerate_videos(directory):
    videos_list = []
    for dirs, _, files in os.walk(directory):
        for file in files:
            videos_list.append(os.path.join(dirs, file))

    return videos_list


def get_time_from_videos(video_path):
    compose = video_path.split(os.sep)
    year_month_day = compose[-1].split(".")[0]
    day, hour = year_month_day.split("_")
    year, month, day = day.split("-")
    hour, minute, second = hour.split("-")
    year_month_day = datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))

    return year_month_day


def get_sequence_of_video_list(directory, interval):
    videos_list = enumerate_videos(directory)

    day_dict = dict(map(lambda x: (x, get_time_from_videos(x)), videos_list))

    videos_filter = list(filter(lambda x: (day_dict[x] >= interval[0] and day_dict[x] < interval[1]), day_dict.keys()))

    videos_filter.sort(key=lambda x: x.split(os.sep)[-1])
    print(len(videos_filter), videos_filter)

    videos_group = []

    for video_id, items in groupby(videos_filter, key=lambda x: x.split(os.sep)[-1]):
        videos_group.append(list(items))

    return videos_group


#videos_group = get_sequence_of_video_list(r"J:\02_吾妻PA_上", (datetime(year=2019, month=9, day=26, hour=12),
#                                                               datetime(year=2019, month=9, day=27, hour=12)))
#
#print(videos_group)