import sys
import os
import time
import json
from datetime import timedelta

from database.sqldatabase import SQLiteDataBase
from database.mysqldatabase import MySQLDataBase

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

use_mysql = True
print(os.path.abspath("../database"))
if not use_mysql:
    database = SQLiteDataBase(database_dir="../database", database_file="edge_matrix_thanh1.db")
else:
    #database = MySQLDataBase(host="localhost", user="Thanh", passwd="Aimesoft", database="2019_10_24", reset_table=False)
    database = MySQLDataBase(host="18.181.144.207", port="3306", user="edge_matrix", passwd="edgematrix", database="edge_matrix_thanh1", reset_table=False)


records = database.get_all_pairs()
print("=================================================================")
for record in records:
    record = list(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, record))
    print(record)
print("=================================================================")
database.close()

#records = list(map(lambda x: tuple(list(x)[:7] + [x[7] - timedelta(hours=9)] + [x[8] - timedelta(hours=9)]), records))

sqlite_database = SQLiteDataBase(database_dir="../database", database_file="edge_matrix_thanh1.db")
sqlite_database.add_pairs(pairs_info=records)

sqlite_database.close()

#for record in records:
#    print(record)

#database = MySQLDataBase(host="18.181.144.207", port="3306", user="edge_matrix", passwd="edgematrix", database="edge_matrix_thanh1", reset_table=False)
#
#database.add_pairs(pairs_info=records)
#
#database.close()