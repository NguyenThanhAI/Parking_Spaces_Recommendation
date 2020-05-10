import sys
import os
import time

from database.sqldatabase import SQLiteDataBase

ROOT_DIR = os.path.abspath("..")
sys.path.append(ROOT_DIR)


print(os.path.abspath("../database"))
sqlite_database = SQLiteDataBase("../database")

try:
    while True:
        records = sqlite_database.get_active_pairs()
        print("=================================================================")
        print(records)
        print("=================================================================")
        time.sleep(5)
except KeyboardInterrupt:
    print("Done and exit")