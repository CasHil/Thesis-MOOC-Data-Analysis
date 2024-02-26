import sqlite3
import os
import subprocess

WORKING_DIR = 'W:/staff-umbrella/gdicsmoocs/Working copy/scripts'
DB_LOCATION = WORKING_DIR + '/thesis_db'

def create_db():
    if os.path.exists(DB_LOCATION):
        os.remove(DB_LOCATION)
    
    conn = sqlite3.connect(DB_LOCATION)
    cur = conn.cursor()

    with open('thesis_schema.sql', 'r') as schema:
            cur.executescript(schema.read())
    
    conn.commit()

    cur.close()
    conn.close()

def insert_data():
    conn = sqlite3.connect(DB_LOCATION)

    cur = conn.cursor()

    for script in os.listdir('./create_insert_scripts'):
        if script.endswith('.py'):
            print("Running script: ", script)
            subprocess.run(["py", "./create_insert_scripts/" + script])
    
    for sql_file in os.listdir(WORKING_DIR):
        if sql_file.endswith('.sql'):
            with open(f"{WORKING_DIR}/{sql_file}", 'r', encoding='utf-8') as insert:
                print(f"Inserting data from {sql_file}")
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
    insert_data()
    delete_sql_files()
