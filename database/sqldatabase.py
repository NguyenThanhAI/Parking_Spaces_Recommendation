import os
import io
import sqlite3
import numpy as np


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)


class SQLiteDataBase(object):
    def __init__(self, database_dir, database_file="edgematrix.db"):
        if not os.path.exists(database_dir):
            os.makedirs(database_dir, exist_ok=True)
        database = os.path.join(database_dir, database_file)
        if not os.path.exists(database):
            self.conn = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            self.create_database()
            self.add_image_links([("parking_ground_SA",
                                   "https://drive.google.com/file/d/1O0eWAWGR6F8x9vLDlHNQ2I4StpjsXeIe/view?usp=sharing"),
                                  ("parking_ground_PA",
                                   "https://drive.google.com/file/d/1FfcOHukOil0w4DlvmE5dI6pLzOmfsQVT/view?usp=sharing")])
        else:
            self.conn = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)

    def create_database(self):
        cursor = self.conn.cursor()
        cursor.execute("""CREATE TABLE PAIRS
                          (CELL_ID INT NOT NULL,
                          VEHICLE_ID INT NOT NULL,
                          CLASS_ID INT  NOT NULL,
                          TYPE_SPACE TEXT NOT NULL,
                          PARKING_GROUND TEXT NOT NULL,
                          CAM TEXT NOT NULL,
                          INACTIVE_STEPS INT NOT NULL,
                          START_TIME  TIMESTAMP NOT NULL,
                          END_TIME TIMESTAMP,
                          PRIMARY KEY (CELL_ID, VEHICLE_ID, CLASS_ID, TYPE_SPACE, PARKING_GROUND, CAM, START_TIME));""")

        cursor.execute("""CREATE TABLE PARKING_SPACES
                          (PARKING_GROUND TEXT NOT NULL,
                          CELL_ID INT NOT NULL,
                          TYPE_SPACE TEXT NOT NULL,
                          COORDINATE ARRAY NOT NULL,
                          PRIMARY KEY (PARKING_GROUND, CELL_ID))""")

        cursor.execute("""CREATE TABLE IMAGE_LINKS
                          (PARKING_GROUND TEXT NOT NULL,
                           URL TEXT,
                           PRIMARY KEY (PARKING_GROUND))""")
        self.conn.commit()

    def add_pairs(self, pairs_info):
        cursor = self.conn.cursor()
        cursor.executemany("INSERT OR REPLACE INTO PAIRS VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", pairs_info)
        self.conn.commit()

    def add_parking_spaces(self, cells_info):
        cursor = self.conn.cursor()
        cursor.executemany("INSERT INTO PARKING_SPACES VALUES (?, ?, ?, ?)", cells_info)
        self.conn.commit()

    def add_image_links(self, links_info):
        cursor = self.conn.cursor()
        cursor.executemany("INSERT OR REPLACE INTO IMAGE_LINKS VALUES (?, ?)", links_info)
        self.conn.commit()

    def get_active_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PAIRS WHERE INACTIVE_STEPS = 0")
        records = cursor.fetchall()
        return records

    def get_non_deleted_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PAIRS WHERE END_TIME = NULL")
        records = cursor.fetchall()
        return records

    def get_deleted_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PAIRS WHERE END_TIME NOT NULL")
        records = cursor.fetchall()
        return records

    def get_all_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PAIRS")
        records = cursor.fetchall()
        return records

    def close(self):
        self.conn.close()
