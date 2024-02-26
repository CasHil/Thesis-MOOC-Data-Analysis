import os

WORKING_DIR = 'W:/staff-umbrella/gdicsmoocs/Working copy/scripts'
DB_LOCATION = WORKING_DIR + '/thesis_db'

def main():
    os.remove(DB_LOCATION)
    print("Database removed")

if __name__ == '__main__':
    main()