import sys
import os
import time

from database.sqldatabase import SQLiteDataBase

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)


print(os.path.abspath("../database"))
sqlite_database = SQLiteDataBase(database_dir="../database", database_file="2019-11-30.db")


try:
    while True:
        records = sqlite_database.get_active_pairs()
        print("=================================================================")
        for record in records:
            record = list(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, record))
            print(record)
        print("=================================================================")
        time.sleep(1)
except KeyboardInterrupt:
    print("Done and exit")