#git add .import sys
#git add .import os
#git add .import time
#git add .from datetime import datetime
#git add .
#git add .from database.sqldatabase import SQLiteDataBase
#git add .from database.mysqldatabase import MySQLDataBase
#git add .
#git add .ROOT_DIR = os.path.abspath("..")
#git add .sys.path.append(ROOT_DIR)
#git add .
#git add .use_mysql = True
#git add .print(os.path.abspath("../database"))
#git add .if not use_mysql:
#git add .    database = SQLiteDataBase(database_dir="../database", database_file="2019-11-24.db")
#git add .else:
#git add .    #database = MySQLDataBase(host="localhost", user="Thanh", passwd="Aimesoft", database="2019_10_24", reset_table=False)
#git add .    database = MySQLDataBase(host="18.181.144.207", port="3306", user="edge_matrix", passwd="edgematrix", database="edge_matrix_bien", reset_table=False)
#git add .
#git add .cursor = database.conn.cursor()
#git add .#cursor.execute("""DELETE FROM PAIRS WHERE START_TIME > '2019-11-27' AND START_TIME < '2019-11-30'""")
#git add .cursor.execute("""DELETE FROM PAIRS""")
#git add .database.conn.commit()
#git add .
#git add .try:
#git add .    records = database.get_all_pairs()
#git add .    print("=================================================================")
#git add .    for record in records:
#git add .        record = list(map(lambda x: int.from_bytes(x, byteorder="little") if isinstance(x, bytes) else x, record))
#git add .        print(record)
#git add .    print("=================================================================")
#git add .    database.close()
#git add .except KeyboardInterrupt:
#git add .    database.close()
#git add .    print("Done and exit")