import sqlite3
import sys
import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

class CYENSDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=CYENSDatabase)

    @staticmethod
    def blob_to_array(blob, dtype, shape=(-1,)):
        if IS_PYTHON3:
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)
        else:
            return np.frombuffer(blob, dtype=dtype).reshape(*shape)

    @staticmethod
    def array_to_blob(array):
        if IS_PYTHON3:
            return array.tostring()
        else:
            return np.getbuffer(array)

    def create_data_table(self):
        CREATE_DATA_TABLE = """CREATE TABLE IF NOT EXISTS image_data (
            image_index INTEGER PRIMARY KEY NOT NULL,
            data BLOB NOT NULL)"""
        self.executescript(CREATE_DATA_TABLE)

    def add_feature_data(self, image_index, values):
        values = np.ascontiguousarray(values, np.float64)
        self.execute(
            "INSERT INTO image_data VALUES (?, ?)",
            (image_index,) + (self.array_to_blob(values),))
        self.commit()

    def get_feature_data(self, image_index, data_length):
        res = self.execute("SELECT data FROM image_data WHERE image_index = " + "'" + str(image_index) + "'")
        res = res.fetchone()[0]
        res = self.blob_to_array(res, np.float64)
        res_rows = int(res.shape[0] / data_length)
        res = res.reshape([res_rows, data_length])
        return res
