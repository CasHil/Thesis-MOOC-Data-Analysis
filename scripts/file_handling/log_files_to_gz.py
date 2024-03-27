import os
import gzip
from dotenv import load_dotenv

load_dotenv()

COURSES = ['EX101x', 'ST1x', 'UnixTx', 'FP101x']
WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')

def log_files_to_gz():
    print("Finding log files...")
    log_files = find_log_files()

    for log_file in log_files:
        print("Processing", log_file)
        with open(log_file, 'rb') as f_in, gzip.open(log_file + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)

    print("Done")

def find_log_files():
    log_files = []
    for root, dirs, files in os.walk(WORKING_DIRECTORY):
        print(f"Processing {root}")
        for file in files:
            if file.endswith(".log") and any(root.split(os.sep)[-1].startswith(course) for course in COURSES):
                print(f"Processing {os.path.join(root, file)}")
                log_files.append(os.path.join(root, file))
    return log_files

def main():
    log_files_to_gz()

if __name__ == '__main__':
    main()