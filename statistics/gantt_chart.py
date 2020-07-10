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

start_time = datetime(year=2019, month=11, day=24, hour=14)

start_time_in_data = min(data, key=lambda x: [7])[7]

print(start_time_in_data)

end_time = datetime(year=2019, month=11, day=24, hour=15, minute=20)

cell_id_to_records = dict(map(lambda x: (x, list(filter(lambda y: y[0] == x, data))), cell_id_list))

cell_id_to_records = dict(map(lambda k: (k, list(map(lambda x: tuple(list(x)[:7] + [start_time] + [x[8]]) if (x[7] < start_time) else x, cell_id_to_records[k]))), cell_id_to_records.keys()))

cell_id_to_records = dict(map(lambda k: (k, list(map(lambda x: tuple(list(x)[:8] + [end_time]) if (x[8] is None) or (x[8] > end_time) else x, cell_id_to_records[k]))), cell_id_to_records.keys()))

print(cell_id_to_records)

df = []

colors = dict(One_car="rgb(0, 102, 192)",
              One_truck="rgb(255, 0, 0)",
              One_bus="rgb(247, 247, 144)",
              One_bicycle="rgb(0, 176, 80)",
              Two_car="rgb(0, 32, 96)",
              Two_truck="rgb(192, 0, 0)",
              Two_bus="rgb(255, 192, 0)",
              Two_bicycle="rgb(84, 130, 53)")

for cell_id in cell_id_to_records:
      start_time_list = list(map(lambda x: (x[7], x[2], 0), cell_id_to_records[cell_id]))

      end_time_list = list(map(lambda x: (x[8], x[2], 1), cell_id_to_records[cell_id]))

      terminal_time_list = (start_time_list + end_time_list)

      terminal_time_list.sort(key=lambda x: x[0])

      print(cell_id, len(terminal_time_list), terminal_time_list)

      num_vehicles = 0

      class_id_list = []

      start_time = None

      for i, terminal_point in enumerate(terminal_time_list):
            if i == 0 or start_time is None:
                  if i == 0:
                        assert terminal_point[2] == 0
                  start_time = terminal_point[0]
                  class_id_list.append(terminal_point[1])
                  num_vehicles += 1
            elif i > 0 and start_time is not None:
                  end_time = terminal_point[0]
                  assert num_vehicles > 0
                  if num_vehicles == 1:
                        if 2 in class_id_list:
                              color = "One_truck"
                        elif 3 in class_id_list and 2 not in class_id_list:
                              color = "One_bus"
                        elif 1 in class_id_list and 2 not in class_id_list and 3 not in class_id_list:
                              color = "One_car"
                        elif 4 in class_id_list and 2 not in class_id_list and 3 not in class_id_list and 1 not in class_id_list:
                              color = "One_bicycle"
                  else:
                        if 2 in class_id_list:
                              color = "Two_truck"
                        elif 3 in class_id_list and 2 not in class_id_list:
                              color = "Two_bus"
                        elif 1 in class_id_list and 2 not in class_id_list and 3 not in class_id_list:
                              color = "Two_car"
                        elif 4 in class_id_list and 2 not in class_id_list and 3 not in class_id_list and 1 not in class_id_list:
                              color = "Two_bicycle"
                  df.append(dict(Task="Cell_id_" + str(cell_id), Start=start_time, Finish=end_time, Resource=color))

                  start_or_end = terminal_point[2]
                  class_id = terminal_point[1]
                  if start_or_end == 0:
                        num_vehicles += 1
                        class_id_list.append(class_id)
                        start_time = end_time
                  else:
                        num_vehicles -= 1
                        print(cell_id, class_id_list, class_id)
                        class_id_list.remove(class_id)
                        if num_vehicles == 0:
                              start_time = None

print(df)

fig = ff.create_gantt(df, colors=colors, index_col="Resource", show_colorbar=True, group_tasks=True, width=10000, height=2000)

fig.show()