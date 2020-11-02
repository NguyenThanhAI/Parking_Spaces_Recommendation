import sys
import os
import time

import os
from datetime import datetime
from datetime import timedelta
import json
import pandas as pd
import numpy as np
from skimage.draw import polygon
import cv2
import plotly
import plotly.figure_factory as ff

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

from database.sqldatabase import SQLiteDataBase
from database.mysqldatabase import MySQLDataBase

use_mysql = False
print(os.path.abspath("../database"))
if not use_mysql:
    database = SQLiteDataBase(database_dir="../database", database_file="edge_matrix_thanh1.db")
else:
    #database = MySQLDataBase(host="localhost", user="Thanh", passwd="Aimesoft", database="2019_10_24", reset_table=False)
    database = MySQLDataBase(host="18.181.144.207", port="3306", user="edge_matrix", passwd="edgematrix", database="edge_matrix_thanh", reset_table=False)

records = database.get_all_pairs()

database.close()

data = []

for record in records:
      record = tuple(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, record))
      data.append(record)

print(data) # Data from database

parking_ground = "parking_ground_PA"

data = list(filter(lambda x: x[4] == parking_ground, data))

sa_cells_mapping = {1: r'小01', 2: r'小02', 3: r'小03', 4: r'小04', 5: r'小05', 6: r'小06', 7: r'小07', 8: r'小08', 9: r'小09', 10: r'小10', 11: r'小11', 12: r'小12', 13: r'小13', 14: r'小14', 15: r'小15', 16: r'小16', 17: r'小17', 18: r'小18', 19: r'小19', 20: r'小20', 21: r'小21', 22: r'小22', 23: r'小23', 24: r'小24', 25: r'小25', 26: r'小26', 27: r'小27', 28: r'小28', 29: r'小29', 30: r'小30', 31: r'小31', 32: r'小32', 33: r'小33', 34: r'小34', 35: r'小35', 36: r'小36', 37: r'小37', 38: r'小38', 39: r'小39', 40: r'小40', 41: r'小41', 42: r'小42', 43: r'小43', 44: r'小44', 45: r'小45', 46: r'小46', 47: r'小47', 48: r'小48', 49: r'小49', 50: r'小50', 51: r'小51', 52: r'小52', 53: r'小53', 54: r'小54', 55: r'小55', 56: r'小56', 57: r'小57', 58: r'小58', 59: r'小59', 60: r'小60', 61: r'小61', 62: r'小62', 63: r'小63', 64: r'小64', 65: r'小65', 66: r'小66', 67: r'小67', 68: r'小68', 69: r'小69', 70: r'小70', 71: r'小71', 72: r'小72', 73: r'大01', 74: r'大02', 75: r'大03', 76: r'大04', 77: r'大05', 78: r'大06', 79: r'大07', 80: r'大08', 81: r'大09', 82: r'大10', 83: r'大11', 84: r'大12', 85: r'大13', 86: r'大14', 87: r'大15', 88: r'大16', 89: r'大17', 90: r'大18', 91: r'大19', 92: r'大20', 93: r'大21', 94: r'大22', 95: r'大23', 96: r'大24', 97: r'小73', 98: r'小74', 99: r'小75', 100: r'小76', 101: r'小77', 102: r'小78', 103: r'小79', 104: r'EV0', 105: r'身02', 106: r'身01', 107: r'二01, 02, 03, 04', 1001: r'外01', 1002: r'外02', 1003: r'外03', 1004: r'外04', 1005: r'外05'}

pa_cells_mapping = {1: r'小01', 2: r'小02', 3: r'小03', 4: r'小04', 5: r'小05', 6: r'小06', 7: r'小07', 8: r'小08', 9: r'小09', 10: r'小10', 11: r'小11', 12: r'小12', 13: r'小13', 14: r'小14', 15: r'小15', 16: r'小16', 17: r'小17', 18: r'小18', 19: r'小19', 20: r'小20', 21: r'小21', 22: r'小22', 23: r'小23', 24: r'小24', 25: r'小25', 26: r'小26', 27: r'小27', 28: r'小28', 29: r'小29', 30: r'小30', 31: r'小31', 32: r'大01', 33: r'大02', 34: r'大03', 35: r'大04', 36: r'大05', 37: r'大06', 38: r'大07', 39: r'大08', 40: r'大09', 41: r'大10', 42: r'大11', 43: r'大12', 44: r'大13', 45: r'大14', 46: r'大15', 47: r'大16', 48: r'大17', 49: r'大18', 50: r'大19', 51: r'大20', 52: r'大21', 53: r'小32', 54: r'小33', 55: r'小34', 56: r'特02', 57: r'特01', 58: r'身01', 59: r'二01,02,03', 60: r'二04', 1001: r'外01', 1002: r'外10', 1003: r'外09', 1004: r'外08', 1005: r'外02', 1006: r'外03', 1007: r'外04', 1008: r'外05', 1009: r'外11', 1010: r'外13', 1011: r'外06, 外07, 外12'}

if parking_ground == "parking_ground_SA":
      cells_mapping = sa_cells_mapping
      cell_number = 112
else:
      cells_mapping = pa_cells_mapping
      cell_number = 68

cell_id_list = list(set(map(lambda x: x[0], data))) # List of cell id

print(cell_id_list)

global_start_time = datetime(year=2019, month=11, day=29, hour=0)

global_end_time = datetime(year=2019, month=11, day=29, hour=12)

#data = list(filter(lambda x: (x[7] >= global_start_time and x[7] <= global_end_time) or (x[8] >= global_start_time and x[8] <= global_end_time) or (x[7] <= global_start_time and x[8] >= global_end_time), data))
data = list(filter(lambda x: (x[7] >= global_start_time and x[7] <= global_end_time) or (x[8] >= global_start_time and x[8] <= global_end_time), data))
#with open("records.txt", "w") as f:
#    for d in data:
#        f.write(str(d) + "\n")
cell_id_to_records = dict(map(lambda x: (x, list(filter(lambda y: y[0] == x, data))), cell_id_list))

cell_id_to_records = dict(map(lambda k: (k, list(map(lambda x: tuple(list(x)[:7] + [global_start_time] + [x[8]]) if (x[7] < global_start_time) else x, cell_id_to_records[k]))), cell_id_to_records.keys()))

cell_id_to_records = dict(map(lambda k: (k, list(map(lambda x: tuple(list(x)[:8] + [global_end_time]) if (x[8] is None) or (x[8] > global_end_time) else x, cell_id_to_records[k]))), cell_id_to_records.keys()))

print(cell_id_to_records)

cell_id_to_heatmap = []

for cell_id in cells_mapping:
    if cell_id not in cell_id_to_records:
        cell_id_to_heatmap.append((cells_mapping[cell_id], str(timedelta(seconds=0)), 0))
        continue

    records_of_cell_id = cell_id_to_records[cell_id]

    records_of_cell_id = list(filter(lambda x: x[7] < x[8], records_of_cell_id))

    if len(records_of_cell_id) == 0:
        cell_id_to_heatmap.append((cells_mapping[cell_id], str(timedelta(seconds=0)), 0))
        continue

    start_time_list = list(map(lambda x: (x[7], 0), records_of_cell_id))

    end_time_list = list(map(lambda x: (x[8], 1), records_of_cell_id))

    terminal_time_list = (start_time_list + end_time_list)

    terminal_time_list.sort(key=lambda x: x[0])

    intervals = []

    start_time = None

    num_vehicles = 0

    for i, terminal_point in enumerate(terminal_time_list):
        if i == 0 or start_time is None:
            if i == 0:
                assert terminal_point[1] == 0
            assert num_vehicles == 0
            start_time = terminal_point[0]
            num_vehicles += 1
        elif i > 0 and start_time is not None:
            if terminal_point[1] == 1:
                num_vehicles -= 1
                if num_vehicles == 0:
                    end_time = terminal_point[0]
                    intervals.append((start_time, end_time))
                    start_time = None
            else:
                num_vehicles += 1

    if len(intervals) > 0:
        total_time = sum(list(map(lambda x: (x[1] - x[0]).total_seconds(), intervals)))
        heatmap = round(total_time * 100 / (global_end_time - global_start_time).total_seconds(), 2)
    else:
        total_time = 0
        heatmap = 0

    print(cells_mapping[cell_id], str(timedelta(seconds=total_time)), heatmap)
    cell_id_to_heatmap.append((cells_mapping[cell_id], str(timedelta(seconds=total_time)), heatmap))

df = pd.DataFrame(cell_id_to_heatmap, columns=["Cell_id", "Parking_time", "Heatmap"])

start_day = global_start_time.strftime("%Y%m%d")
end_day = global_end_time.strftime("%Y%m%d")

save_dir = r"C:\Users\Thanh\Downloads\heatmap"

if start_day == end_day:
    save_dir = os.path.join(save_dir, parking_ground.split("_")[-1], global_end_time.strftime("%Y%m%d"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    csvfilename = "cells_heatmap" + "-" + str(global_start_time.hour).zfill(2) + str(global_end_time.hour).zfill(2) + "-" + end_day + "-" + parking_ground.split("_")[-1] + ".csv"
    imagefilename = "cells_heatmap" + "-" + str(global_start_time.hour).zfill(2) + str(global_end_time.hour).zfill(2) + "-" + end_day + "-" + parking_ground.split("_")[-1] + ".png"
else:
    save_dir = os.path.join(save_dir, parking_ground.split("_")[-1], global_start_time.strftime("%Y%m%d"))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    csvfilename = "cells_heatmap" + "-" + str(global_start_time.hour).zfill(2) + "-" + start_day + "-" + str(global_end_time.hour).zfill(2) + "-" + end_day + "-" + parking_ground.split("_")[-1] + ".csv"
    imagefilename = "cells_heatmap" + "-" + str(global_start_time.hour).zfill(2) + "-" + start_day + "-" + str(global_end_time.hour).zfill(2) + "-" + end_day + "-" + parking_ground.split("_")[-1] + ".png"

#filename = "cells_heatmap" + "-" + str(global_start_time.hour).zfill(2) + "-" + start_day + "-" + str(global_end_time.hour).zfill(2) + "-" + end_day + "-" + parking_ground.split("_")[-1] + ".csv"

df.to_csv(os.path.join(save_dir, csvfilename), encoding="utf_8_sig")

cells_heatmap = dict(map(lambda x: (x[0], x[2]), cell_id_to_heatmap))

image_ground_dir = r"C:\Users\Thanh\Downloads\Parking_ground_images_and_db_ver_2"
label_file_path = "../parking_spaces_data/parking_spaces_unified_id_segmen_in_ground.json"

if parking_ground == "parking_ground_SA":
    file_name = "cropped_SA.jpg"
else:
    file_name = "cropped_PA.jpg"

file_path = os.path.join(image_ground_dir, file_name)
img = cv2.imread(file_path)

with open(label_file_path, "r") as f:
    json_label = json.load(f)

unified_id_to_polygons = json_label

color_dict = {(0., 1.): (255, 255, 255),
              (1., 20.): (153, 51, 102),
              (20., 40.): (255, 0, 0),
              (40., 60.): (0, 255, 0),
              (60., 80.): (0, 255, 255),
              80.: (0, 0, 255)}

for unified_id in unified_id_to_polygons[parking_ground]:
    if int(unified_id) not in cells_heatmap:
        cells_heatmap[int(unified_id)] = 0.
    segment = unified_id_to_polygons[parking_ground][unified_id]["positions"]
    segment = np.array(segment, dtype=np.uint16).reshape(-1, 2)
    cc, rr = segment.T
    rr, cc = polygon(rr, cc)
    percent = cells_heatmap[cells_mapping[int(unified_id)]]
    for percent_time in color_dict:
        if isinstance(percent_time, float):
            if percent_time == 0.:
                if percent <= percent_time:
                    color = color_dict[percent_time]
                    break
                else:
                    continue
            else:
                if percent > percent_time:
                    color = color_dict[percent_time]
                    break
                else:
                    continue
        else:
            if percent >= percent_time[0] and percent < percent_time[1]:
                color = color_dict[percent_time]
                break
            else:
                continue
    img[rr, cc] = color
    center_x, center_y = np.mean(segment, axis=0).astype(np.uint16)
    segment = segment.tolist()
    color = (0, 0, 0)
    for j, point in enumerate(segment):
        x1, y1, = point
        if j < len(segment) - 1:
            x2, y2 = segment[j + 1]
        else:
            x2, y2 = segment[0]

        cv2.line(img, (x1, y1), (x2, y2), color=color, thickness=1)

    cv2.putText(img, "{}".format(unified_id), (center_x, center_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), thickness=1)

cv2.imshow("{}".format("".join([parking_ground, "_heatmap"])), img)
cv2.waitKey(0)

cv2.imwrite(os.path.join(save_dir, imagefilename), img)
