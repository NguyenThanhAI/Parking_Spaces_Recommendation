import sys
import os
import time
import json

from tqdm import tqdm

from database.sqldatabase import SQLiteDataBase
from database.mysqldatabase import MySQLDataBase

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

sqlite_database = SQLiteDataBase(database_dir="../database", database_file="edge_matrix_thanh1.db")

mysql_database = MySQLDataBase(host="18.181.144.207", port="3306", user="edge_matrix", passwd="edgematrix", database="edge_matrix_thanh1", reset_table=False)


records = sqlite_database.get_all_pairs()
for record in tqdm(records):
    record = [tuple(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, record))]
    print(record)
    mysql_database.add_pairs(pairs_info=record)

with open("../parking_spaces_data/parking_spaces_unified_id_segmen_in_ground.json", "r") as f:
    cell_infos = json.load(f)

cell_infos_list = []
for parking_ground in cell_infos:
    for cell_id in cell_infos[parking_ground]:
        cell_infos_list.append((parking_ground, int(cell_id), cell_infos[parking_ground][cell_id]["type_space"], str(cell_infos[parking_ground][cell_id]["positions"])))

mysql_database.add_parking_spaces(cells_info=cell_infos_list)

sqlite_database.close()
mysql_database.close()