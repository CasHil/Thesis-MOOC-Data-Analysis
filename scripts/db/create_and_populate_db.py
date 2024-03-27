import sqlite3
import os
import subprocess

WORKING_DIR = 'W:/staff-umbrella/gdicsmoocs/Working copy/scripts'

def create_db(db_name: str) -> None:
    db_location = f"{WORKING_DIR}/{db_name}"
    if os.path.exists(db_location):
        os.remove(db_location)
    
    conn = sqlite3.connect(db_location)
    cur = conn.cursor()

    with open('thesis_schema.sql', 'r') as schema:
            cur.executescript(schema.read())
    
    conn.commit()

    cur.close()
    conn.close()

def insert_demographic_data(db_name: str, directory: str, directory_files: list[str]) -> None:
    db_location = f"{WORKING_DIR}/{db_name}"

    conn = sqlite3.connect(db_location)

    cur = conn.cursor()

    for script in directory_files:
        if script.endswith('.py'):
            print("Running script: ", script)
            subprocess.run(["py",  f"{directory}/{script}"])
    
    for sql_file in os.listdir(WORKING_DIR):
        if sql_file.endswith('.sql'):
            with open(f"{WORKING_DIR}/{sql_file}", 'r', encoding='utf-8') as insert:
                print(f"Inserting data from {sql_file} into {db_name}...")
                cur.executescript(insert.read())
                    
    conn.commit()

    cur.close()
    conn.close()

def delete_sql_files():
    for sql_file in os.listdir(WORKING_DIR):
        if sql_file.endswith('.sql'):
            os.remove(f'{WORKING_DIR}/{sql_file}')

if __name__ == '__main__':
    create_db()
    insert_demographic_data()
    # delete_sql_files()
