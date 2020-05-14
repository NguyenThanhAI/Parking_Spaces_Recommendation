from itertools import groupby
import datetime
import plotly.figure_factory as ff
from database.sqldatabase import SQLiteDataBase
from code_timing_profiling.timing import timethis


def convert_bytes_to_int(records: list):
    return list(map(lambda y: tuple(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, y)), records))


def gather_time_intervals(records: list):
    return list(map(lambda x: (x[6], x[7]), records))


def gather_class_vehicle_id_type_space(records: list):
    return list(map(lambda x: (x[1], x[2], x[3]), records))


def get_records_grouped_by_unified_id(records: list):
    info_dicts = {}
    for unified_id, grouped in groupby(sorted(records, key=lambda x: x[0]), key=lambda x: x[0]):
        info_dicts[unified_id] = {}
        grouped = list(grouped)
        time_intervals = gather_time_intervals(grouped)
        info_dicts[unified_id]["time_intervals"] = time_intervals
        infos = gather_class_vehicle_id_type_space(grouped)
        info_dicts[unified_id]["infos"] = infos
        #print(unified_id, infos, time_intervals)

    return info_dicts


def gather_unified_id_type_space(records: list):
    return list(map(lambda x: (x[0], x[3]), records))


def gather_class_id(records: list):
    return list(map(lambda x: x[2], records))


def get_records_grouped_by_vehicle_id(records: list):
    info_dicts = {}
    for vehicle_id, grouped in groupby(sorted(records, key=lambda x: x[1]), key=lambda x: x[1]):
        info_dicts[vehicle_id] = {}
        grouped = list(grouped)
        time_intervals = gather_time_intervals(grouped)
        info_dicts[vehicle_id]["time_intervals"] = time_intervals
        infos = gather_unified_id_type_space(grouped)
        info_dicts[vehicle_id]["infos"] = infos
        class_id = gather_class_id(records)[0]
        info_dicts[vehicle_id]["class_id"] = class_id
        #print(vehicle_id, infos, time_intervals)

    return info_dicts


@timethis
def get_vehicle_id_infos(records: list):
    info_dicts = get_records_grouped_by_vehicle_id(records)
    time_interval = dict(map(lambda x: (x, (min(info_dicts[x]["time_intervals"], key=lambda y: y[0])[0], max(list(filter(lambda z: z[1] is not None, info_dicts[x]["time_intervals"])), key=lambda y: y[1])[1] if any(list(map(lambda x: x[1] is not None, info_dicts[x]["time_intervals"]))) else None)), info_dicts.keys()))
    info_dicts = dict(map(lambda x: (x, {"Class_id": info_dicts[x]["class_id"], "Parking_time": (time_interval[x][1] - time_interval[x][0]) if time_interval[x][1] else "Parking", "Start_time": time_interval[x][0], "End_time": time_interval[x][1], "Cells_info": info_dicts[x]["infos"]}), info_dicts.keys()))

    return info_dicts


def find_union_of_time_intervals(intervals: list, end_time=datetime.datetime(year=2019, month=11, day=30, hour=11, minute=0, second=0)):
    intervals = sorted(intervals, key=lambda x: x[0])
    result = []
    (start_candidate, stop_candidate) = intervals[0]
    if not stop_candidate:
        stop_candidate = end_time
    for (start, stop) in intervals[1:]:
        if not stop:
            stop = end_time
        if start <= stop_candidate:
            stop_candidate = max(stop, stop_candidate)
        else:
            result.append((start_candidate, stop_candidate))
            (start_candidate, stop_candidate) = (start, stop)
    result.append((start_candidate, stop_candidate))
    return result


def create_gantt_chart_plot(records: list, end_time: datetime.datetime=datetime.datetime(year=2019, month=11, day=30, hour=10, minute=10, second=0)):
    df = list(map(lambda x: dict(Task="Cell_id " + str(x[0]), Start=x[6].strftime("%Y-%m-%d %H:%M:%S"), Finish=x[7].strftime("%Y-%m-%d %H:%M:%S") if x[7] is not None else end_time, Type_Space="small"), sorted(records, key=lambda x: x[0])))

    colors = dict(small="rgb(0, 0, 255",
                  big="rgb(255, 0, 0")

    fig = ff.create_gantt(df, colors=colors, index_col="Type_Space", title="Gantt Chart", show_colorbar=True,
                          bar_width=0.1, showgrid_x=True, showgrid_y=True, group_tasks=True, height=2000, width=2000)

    fig.show()


@timethis
def get_heatmap(records: list, start_time=None, end_time=None):
    if not start_time:
        start_time = min(records, key=lambda x: x[6])[6]
    if not end_time:
        end_time = max(list(filter(lambda x: x[7] is not None, records)))[7] if any(map(lambda x: x[7] is not None, records)) else None
    #print(start_time, end_time, type(start_time), type(end_time))
    info_dicts = get_records_grouped_by_unified_id(records)
    union_intervals = dict(map(lambda x: (x, find_union_of_time_intervals(info_dicts[x]["time_intervals"], end_time=end_time)), info_dicts.keys()))
    union_intervals = dict(map(lambda x: (x, sum(list(map(lambda y: (y[1] - y[0]).total_seconds(), union_intervals[x])))), union_intervals.keys()))
    #print(union_intervals)
    total_time = (end_time - start_time).total_seconds()
    heatmap = dict(map(lambda x: (x, round((union_intervals[x] / total_time) * 100., 2)), union_intervals.keys()))
    #print(heatmap)
    return heatmap
