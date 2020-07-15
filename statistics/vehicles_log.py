import sys
import os
import time

import os
from datetime import datetime
from itertools import groupby
import plotly
import plotly.figure_factory as ff

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

from database.sqldatabase import SQLiteDataBase

database = SQLiteDataBase(database_dir="../database", database_file="2019-11-24.db")

records = database.get_all_pairs()

data = []

for record in records:
      record = tuple(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, record))
      data.append(record)

print(data) # Data from database

parking_ground = "parking_ground_SA"

data = list(filter(lambda x: x[4] == parking_ground, data))

print(data)

cam_list = list(set(map(lambda x: x[5], data)))

cam_list.sort()
print(cam_list)

data.sort(key=lambda x: x[5])

cam_to_records = {}

for cam, items in groupby(data, key=lambda x: x[5]):
    cam_to_records[cam] = list(items)

print(cam_to_records)

cam_veh_id_records = {}

for cam in cam_to_records:
    records = cam_to_records[cam]

    records.sort(key=lambda x: x[1])

    cam_veh_id_records[cam] = {}

    for veh_id, items in groupby(data, key=lambda x: x[1]):
        cam_veh_id_records[cam][veh_id] = list(items)

print(cam_veh_id_records)

for cam in cam_veh_id_records:
    for veh_id in cam_veh_id_records[cam]:
        record = []
        records = cam_veh_id_records[cam][veh_id]
        if len(records) > 1:
            cells_id = tuple(map(lambda x: x[0], records))
            class_id = records[0][2]
            type_space = tuple(map(lambda x: x[3], records))
            parking_ground = records[0][4]
            cam = records[0][5]
            inactive_steps = min(records, key=lambda x: x[6])
            start_time = min(records, key=lambda x: x[7])
            end_time = max(records, key=lambda x: x[8])
        else:
            cells_id = records[0]
            class_id = records[2]
            start_time = records[7]
            end_time = records[8]
