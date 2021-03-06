import os
import mysql.connector


class MySQLDataBase(object):
    def __init__(self, host, user, passwd, database, port=None, reset_table=False):
        if port is None:
            try:
                self.conn = mysql.connector.connect(host=host,
                                                    user=user,
                                                    passwd=passwd,
                                                    database=database)
            except:
                self.conn = mysql.connector.connect(host=host,
                                                    user=user,
                                                    passwd=passwd)
                self.create_database(database)
        else:
            try:
                self.conn = mysql.connector.connect(host=host,
                                                    port=port,
                                                    user=user,
                                                    passwd=passwd,
                                                    database=database)
            except:
                self.conn = mysql.connector.connect(host=host,
                                                    port=port,
                                                    user=user,
                                                    passwd=passwd)
                self.create_database(database)

        if reset_table:
            self.reset_table()

        self.create_tables()
        self.add_image_links([("parking_ground_SA", "https://drive.google.com/file/d/1O0eWAWGR6F8x9vLDlHNQ2I4StpjsXeIe/view?usp=sharing"),
                              ("parking_ground_PA", "https://drive.google.com/file/d/1FfcOHukOil0w4DlvmE5dI6pLzOmfsQVT/view?usp=sharing")])

    def create_database(self, database):
        cursor = self.conn.cursor()
        cursor.execute("CREATE DATABASE %s" % (database))
        cursor.execute("USE %s" % (database))

    def reset_table(self):
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS PAIRS")
        cursor.execute("DROP TABLE IF EXISTS PARKING_SPACES")

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""SET sql_mode = '';""")
        cursor.execute("""CREATE TABLE IF NOT EXISTS PAIRS
                          (CELL_ID INT NOT NULL,
                          VEHICLE_ID INT NOT NULL,
                          CLASS_ID INT  NOT NULL,
                          TYPE_SPACE VARCHAR(8) NOT NULL,
                          PARKING_GROUND VARCHAR(20) NOT NULL,
                          CAM VARCHAR(5) NOT NULL,
                          INACTIVE_STEPS INT NOT NULL,
                          START_TIME TIMESTAMP NOT NULL,
                          END_TIME TIMESTAMP,
                          PRIMARY KEY (CELL_ID, VEHICLE_ID, CLASS_ID, TYPE_SPACE, PARKING_GROUND, CAM, START_TIME));""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS PARKING_SPACES
                          (PARKING_GROUND VARCHAR(20) NOT NULL,
                          CELL_ID INT NOT NULL,
                          TYPE_SPACE VARCHAR(8) NOT NULL,
                          COORDINATE LONGTEXT NOT NULL,
                          PRIMARY KEY (PARKING_GROUND, CELL_ID));""")

        cursor.execute("""CREATE TABLE IF NOT EXISTS IMAGE_LINKS
                          (PARKING_GROUND VARCHAR(20) NOT NULL,
                           URL VARCHAR(512),
                           PRIMARY KEY (PARKING_GROUND));""")

    def add_pairs(self, pairs_info):
        cursor = self.conn.cursor()
        cursor.executemany("REPLACE INTO PAIRS (CELL_ID, VEHICLE_ID, CLASS_ID, TYPE_SPACE, PARKING_GROUND, CAM, INACTIVE_STEPS, START_TIME, END_TIME) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)", pairs_info)
        self.conn.commit()
        cursor.close()

    def add_parking_spaces(self, cells_info):
        cursor = self.conn.cursor()
        cursor.executemany("REPLACE INTO PARKING_SPACES (PARKING_GROUND, CELL_ID, TYPE_SPACE, COORDINATE) VALUES (%s, %s, %s, %s)", cells_info)
        self.conn.commit()
        cursor.close()

    def add_image_links(self, links_info):
        cursor = self.conn.cursor()
        cursor.executemany("REPLACE INTO IMAGE_LINKS (PARKING_GROUND, URL) VALUES (%s, %s)", links_info)
        self.conn.commit()
        cursor.close()

    def get_active_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PAIRS WHERE INACTIVE_STEPS = 0")
        records = cursor.fetchall()
        cursor.close()
        return records

    def get_non_deleted_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PAIRS WHERE END_TIME IS NULL")
        records = cursor.fetchall()
        cursor.close()
        return records

    def get_deleted_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PAIRS WHERE END_TIME IS NOT NULL")
        records = cursor.fetchall()
        cursor.close()
        return records

    def get_all_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM PAIRS")
        records = cursor.fetchall()
        cursor.close()
        return records

    def close(self):
        self.conn.close()