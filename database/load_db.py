import pickle
import pandas as pd


with open("2019-11-30.pkl", "rb") as f:
    pairs = pickle.load(f)

veh_pairs = dict(sorted(pairs.items(), key=lambda x: x[1].vehicle_id))

for pair in veh_pairs:
    print(veh_pairs[pair])

uid_pairs = dict(sorted(pairs.items(), key=lambda x: x[1].unified_id))

for pair in uid_pairs:
    print(uid_pairs[pair])

data = list(map(lambda x: (x[1].unified_id, x[1].vehicle_id, x[1].birth_time.strftime("%Y-%m-%d %H:%M:%S"), x[1].end_time.strftime("%Y-%m-%d %H:%M:%S") if x[1].end_time else None), uid_pairs.items()))

data = pd.DataFrame(data=data, index=None, columns=["Cell ID", "Vehicle ID", "Start time", "End time"])

data.to_csv("2019-11-30.csv")