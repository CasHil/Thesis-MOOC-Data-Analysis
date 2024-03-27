import os

WORKING_DIR = 'W:/staff-umbrella/gdicsmoocs/Working copy/scripts/'

def remove_db(db_name: str) -> None:
    db_location = WORKING_DIR + db_name
    os.remove(db_location)
    print(f"Database {db_location} removed")

