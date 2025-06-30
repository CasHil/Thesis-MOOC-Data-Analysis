import sqlite3
import os
import json
from dotenv import load_dotenv

load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')


def dump_db_to_json():
    conn = sqlite3.connect(MOOC_DB_LOCATION)

    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cur.fetchall()

    for table in tables:
        table_name = table[0]

        cur.execute(f"SELECT * FROM {table_name};")
        rows = cur.fetchall()

        rows_as_dict = [dict(row) for row in rows]

        json_file_path = os.path.join(WORKING_DIRECTORY, f"{table_name}.json")

        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(rows_as_dict, json_file, ensure_ascii=False, indent=4)

    cur.close()
    conn.close()


if __name__ == '__main__':
    dump_db_to_json()
