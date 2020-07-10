import sys
import os
import time

import os
from datetime import datetime
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

cell_id_list = list(set(map(lambda x: x[0], data))) # List of cell id

print(cell_id_list)

global_start_time = datetime(year=2019, month=11, day=24, hour=14)

start_time_in_data = min(data, key=lambda x: [7])[7]

print(start_time_in_data)

global_end_time = datetime(year=2019, month=11, day=24, hour=15, minute=20)

cell_id_to_records = dict(map(lambda x: (x, list(filter(lambda y: y[0] == x, data))), cell_id_list))

cell_id_to_records = dict(map(lambda k: (k, list(map(lambda x: tuple(list(x)[:7] + [global_start_time] + [x[8]]) if (x[7] < global_start_time) else x, cell_id_to_records[k]))), cell_id_to_records.keys()))

cell_id_to_records = dict(map(lambda k: (k, list(map(lambda x: tuple(list(x)[:8] + [global_end_time]) if (x[8] is None) or (x[8] > global_end_time) else x, cell_id_to_records[k]))), cell_id_to_records.keys()))

print(cell_id_to_records)

for cell_id in cell_id_to_records:
    start_time_list = list(map(lambda x: (x[7], 0), cell_id_to_records[cell_id]))

    end_time_list = list(map(lambda x: (x[8], 1), cell_id_to_records[cell_id]))

    terminal_time_list = (start_time_list + end_time_list)

    terminal_time_list.sort(key=lambda x: x[0])

    intervals = []

    start_time = None

    for i, terminal_point in enumerate(terminal_time_list):
        if i == 0 or start_time is None:
            if i == 0:
                assert terminal_point[1] == 0
            start_time = terminal_point[0]
        elif i > 0 and start_time is not None:
            if terminal_point[1] == 1:
                end_time = terminal_point[0]
                intervals.append((start_time, end_time))
                start_time = None
            else:
                continue

    if len(intervals) > 0:
        heatmap = round(sum(list(map(lambda x: (x[1] - x[0]).total_seconds(), intervals))) * 100 / (global_end_time - global_start_time).total_seconds(), 3)
    else:
        heatmap = 0

    print(heatmap)
