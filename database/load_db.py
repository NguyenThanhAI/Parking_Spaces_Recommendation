import sys
import os
import time

from database.sqldatabase import SQLiteDataBase
from database.mysqldatabase import MySQLDataBase

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)

use_mysql = False
print(os.path.abspath("../database"))
if not use_mysql:
    database = SQLiteDataBase(database_dir="../database", database_file="2019-10-24.db")
else:
    database = MySQLDataBase(host="localhost", user="Thanh", passwd="Aimesoft", database="2019_10_24", reset_table=False)


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
    print("Done and exit")
