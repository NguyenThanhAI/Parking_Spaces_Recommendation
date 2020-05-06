import pickle
import pandas as pd


with open("2019-11-30.pkl", "rb") as f:
    pairs = pickle.load(f)

veh_pairs = sorted(pairs, key=lambda x: x.vehicle_id)

for pair in veh_pairs:
    print(pair)

uid_pairs = sorted(pairs, key=lambda x: x.unified_id)

for pair in uid_pairs:
    print(pair)

data = list(map(lambda x: (x.unified_id, x.vehicle_id, x.birth_time.strftime("%Y-%m-%d %H:%M:%S"), x.end_time.strftime("%Y-%m-%d %H:%M:%S") if x.end_time else None), uid_pairs))

data = pd.DataFrame(data=data, index=None, columns=["Cell ID", "Vehicle ID", "Start time", "End time"])

data.to_csv("2019-11-30.csv")