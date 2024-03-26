import os
import gzip

def find_log_files(base_path, prefixes):
    log_files = []
    for root, dirs, files in os.walk(base_path):
        print(f"Processing {root}")
        for file in files:
            if file.endswith(".log") and any(root.split(os.sep)[-1].startswith(prefix) for prefix in prefixes):
                print(f"Processing {os.path.join(root, file)}")
                log_files.append(os.path.join(root, file))
    return log_files

base_path = 'W:/staff-umbrella/gdicsmoocs/Working copy'
prefixes = ["EX101x", "ST1x", "UnixTx", "FP101x"]

print("Finding log files...")
log_files = find_log_files(base_path, prefixes)

for log_file in log_files:
    print("Processing", log_file)
    with open(log_file, 'rb') as f_in, gzip.open(log_file + '.gz', 'wb') as f_out:
        f_out.writelines(f_in)

print("Done")