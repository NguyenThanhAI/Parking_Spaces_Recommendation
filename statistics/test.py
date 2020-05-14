import datetime
import pandas as pd
from database.sqldatabase import SQLiteDataBase
from statistics.utils import convert_bytes_to_int, get_records_grouped_by_unified_id, get_records_grouped_by_vehicle_id, \
    find_union_of_time_intervals, get_heatmap, create_gantt_chart_plot, get_vehicle_id_infos


sqlite_database = SQLiteDataBase(database_dir="../database", database_file="2019-11-30.db")
records = sqlite_database.get_all_pairs()

records = convert_bytes_to_int(records)
print(records)

info_dicts = get_records_grouped_by_unified_id(records)
print(info_dicts)

info_dicts = get_records_grouped_by_vehicle_id(records)
print(info_dicts)

vehicles_info = get_vehicle_id_infos(records)
print(vehicles_info)
vehicles_df = pd.DataFrame.from_dict(vehicles_info, orient="index")
print(vehicles_df)
vehicles_df.to_csv("vehicle_info.csv")

intervals = [(40, 64), (3, 9), (3, 10), (4, 15), (5, 21), (8, 24), (33, 40), (35, 61)]
#intervals = [(datetime.datetime(2019, 11, 30, 10, 0, 1), datetime.datetime(2019, 11, 30, 10, 0, 11)), (datetime.datetime(2019, 11, 30, 10, 0, 40), None), (datetime.datetime(2019, 11, 30, 10, 0, 11), None), (datetime.datetime(2019, 11, 30, 10, 0, 11), None)]
print(find_union_of_time_intervals(intervals, end_time=100))

heatmap = get_heatmap(records)
print(heatmap)
heatmap_df = pd.DataFrame.from_dict(heatmap, orient="index", columns=["heatmap"])
heatmap_df.to_csv("cells_heatmap.csv")

create_gantt_chart_plot(records)
