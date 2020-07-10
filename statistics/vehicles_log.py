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

for cam in cam_to_records:
    records = cam_to_records[cam]

    records.sort(key=lambda x: x[1])

    for veh_id, items in groupby(data, key=lambda x: x[1]):
        print(cam, veh_id, list(items))
