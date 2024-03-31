import sqlite3
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

SCRIPTS_DIRECTORY = os.getenv('SCRIPTS_DIRECTORY')
MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')
MOOC_DB_DIRECTORY = os.getenv('MOOC_DB_DIRECTORY')


def create_db() -> None:
    if os.path.exists(MOOC_DB_LOCATION):
        os.remove(MOOC_DB_LOCATION)
    else:
        os.makedirs(MOOC_DB_DIRECTORY)

    conn = sqlite3.connect(MOOC_DB_LOCATION)
    conn.execute("PRAGMA foreign_keys = ON;")
    cur = conn.cursor()

    with open('mooc_db_schema.sql', 'r') as schema:
        cur.executescript(schema.read())

    print("Database created successfully.")

    conn.commit()

    cur.close()
    conn.close()


def insert_demographic_data() -> None:

    conn = sqlite3.connect(MOOC_DB_LOCATION)

    cur = conn.cursor()

    subprocess.run(["py",  './create_insert_scripts.py'])

    for sql_file in os.listdir(MOOC_DB_DIRECTORY):
        if sql_file.endswith('.sql'):
            with open(os.path.join(MOOC_DB_DIRECTORY, sql_file), 'r', encoding='utf-8') as insert:
                print(f"Inserting data from {sql_file}...")
                cur.executescript(insert.read())

    conn.commit()

    cur.close()
    conn.close()


def insert_metadata() -> None:
    subprocess.run(["py", "./insert_metadata.py"])


def delete_sql_files():
    for sql_file in os.listdir(SCRIPTS_DIRECTORY):
        if sql_file.endswith('.sql'):
            os.remove(f'{SCRIPTS_DIRECTORY}/{sql_file}')


if __name__ == '__main__':
    recreate_db_response = input("(Re)create database? (Y/N): ")
    if recreate_db_response == 'Y':
        create_db()

    insert_metadata()
    insert_demographic_data()
    # Optional
    # delete_sql_files()
