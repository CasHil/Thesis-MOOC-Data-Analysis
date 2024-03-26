import os
import shutil
import time

top_folder_name = input("Please enter the name for the top-level folder: ")
downloads_path = 'C:/Users/Casper/Downloads'
destination_path = os.path.join(downloads_path, top_folder_name)

now = time.time()
cycle_charts = []
webdata_export_file = None
for file in os.listdir(downloads_path):
    file_path = os.path.join(downloads_path, file)
    if now - os.path.getmtime(file_path) <= 1000:
        if file.startswith("cycleChart"):
            cycle_charts.append((file, os.path.getmtime(file_path)))
        elif "webdata_export" in file:
            webdata_export_file = file

if len(cycle_charts) % 3 != 0:
    raise ValueError("The total number of cycleChart files is not divisible by 3.")

cycle_charts.sort(key=lambda x: x[1])
cycle_chart_filenames = [x[0] for x in cycle_charts]

third = len(cycle_chart_filenames) // 3
groups = [cycle_chart_filenames[:third], cycle_chart_filenames[third:2*third], cycle_chart_filenames[2*third:]]

subfolders = ["Men and women", "Men", "Women"]
os.makedirs(destination_path, exist_ok=True)
for subfolder in subfolders:
    os.makedirs(os.path.join(destination_path, subfolder), exist_ok=True)

for i, group in enumerate(groups):
    for file in group:
        shutil.move(os.path.join(downloads_path, file), os.path.join(destination_path, subfolders[i], file))

if webdata_export_file:
    shutil.move(os.path.join(downloads_path, webdata_export_file), destination_path)

print("Files have been organized successfully.")
