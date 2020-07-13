import sys
import os
import time

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
    database = MySQLDataBase(host="18.181.144.207", port="3306", user="edge_matrix", passwd="edgematrix", database="edge_matrix_thanh", reset_table=False)


try:
    while True:
        records = database.get_all_pairs()
        print("=================================================================")
        for record in records:
            record = list(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, record))
            print(record)
        print("=================================================================")
        time.sleep(60)
except KeyboardInterrupt:
    database.close()
    print("Done and exit")
