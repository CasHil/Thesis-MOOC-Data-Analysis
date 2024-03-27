import sqlite3
import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

SCRIPTS_DIRECTORY = os.getenv('SCRIPTS_DIRECTORY')
MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')

def create_db() -> None:
    if os.path.exists(MOOC_DB_LOCATION):
        os.remove(MOOC_DB_LOCATION)
    
    conn = sqlite3.connect(MOOC_DB_LOCATION)
    cur = conn.cursor()

    with open('thesis_schema.sql', 'r') as schema:
            cur.executescript(schema.read())
    
    conn.commit()

    cur.close()
    conn.close()

def insert_demographic_data() -> None:

    conn = sqlite3.connect(MOOC_DB_LOCATION)

    cur = conn.cursor()

    insert_script_directory = "./create_insert_scripts"
    insert_script_directory_files = os.listdir(insert_script_directory)
    
    for script in insert_script_directory_files:
        if script.endswith('.py'):
            print("Running script: ", script)
            subprocess.run(["py",  f"{insert_script_directory}/{script}"])
    
    for sql_file in os.listdir(SCRIPTS_DIRECTORY):
        if sql_file.endswith('.sql'):
            with open(f"{SCRIPTS_DIRECTORY}/{sql_file}", 'r', encoding='utf-8') as insert:
                print(f"Inserting data from {sql_file}...")
                cur.executescript(insert.read())
                    
    conn.commit()

    cur.close()
    conn.close()

def delete_sql_files():
    for sql_file in os.listdir(SCRIPTS_DIRECTORY):
        if sql_file.endswith('.sql'):
            os.remove(f'{SCRIPTS_DIRECTORY}/{sql_file}')

if __name__ == '__main__':
    create_db()
    insert_demographic_data()
    # delete_sql_files()
