import sys
import os
import time
import json

from database.sqldatabase import SQLiteDataBase
from database.mysqldatabase import MySQLDataBase

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

use_mysql = True
print(os.path.abspath("../database"))
if not use_mysql:
    database = SQLiteDataBase(database_dir="../database", database_file="2019-11-24.db")
else:
    #database = MySQLDataBase(host="localhost", user="Thanh", passwd="Aimesoft", database="2019_10_24", reset_table=False)
    database = MySQLDataBase(host="18.181.144.207", port="3306", user="edge_matrix", passwd="edgematrix", database="edge_matrix_bien", reset_table=False)

with open("../parking_spaces_data/parking_spaces_unified_id_segmen_in_ground.json", "r") as f:
    cell_infos = json.load(f)

cell_infos_list = []
for parking_ground in cell_infos:
    for cell_id in cell_infos[parking_ground]:
        cell_infos_list.append((parking_ground, int(cell_id), cell_infos[parking_ground][cell_id]["type_space"], str(cell_infos[parking_ground][cell_id]["positions"])))

database.add_parking_spaces(cells_info=cell_infos_list)

sqlite_database = SQLiteDataBase(database_dir="../database", database_file="2019-11-24.db")

records = sqlite_database.get_all_pairs()
for i, record in enumerate(records):
    if i == 2:
        database.close()
        break
    record = [tuple(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, record))]
    print(record)
    database.add_pairs(pairs_info=record)
