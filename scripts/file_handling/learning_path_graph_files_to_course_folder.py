import os
import shutil
import time


def organize_learning_path_graph_files():
    top_folder_name = input("Please enter the name for the top-level folder: ")
    downloads_path = 'D:/Downloads'
    destination_path = os.path.join(downloads_path, top_folder_name)

    now = time.time()
    cycle_charts = []
    webdata_export_file = None
    for file in os.listdir(downloads_path):
        file_path = os.path.join(downloads_path, file)
        if now - os.path.getmtime(file_path) <= 1000:
            if file.startswith("cycleChart"):
                cycle_charts.append(file)
            elif "webdata_export" in file:
                webdata_export_file = file

    if len(cycle_charts) % 3 != 0:
        raise ValueError(
            "The total number of cycleChart files is not divisible by 3.")

    cycle_charts.sort(key=lambda x: int(
        x.split('(')[-1].split(')')[0]) if '(' in x else 0)

    third = len(cycle_charts) // 3
    groups = [cycle_charts[:third],
              cycle_charts[third:2*third], cycle_charts[2*third:]]

    subfolders = ["Men and women", "Men", "Women"]
    os.makedirs(destination_path, exist_ok=True)
    for subfolder in subfolders:
        os.makedirs(os.path.join(destination_path, subfolder), exist_ok=True)

    for i, group in enumerate(groups):
        for file in group:
            shutil.move(os.path.join(downloads_path, file),
                        os.path.join(destination_path, subfolders[i], file))

    if webdata_export_file:
        shutil.move(os.path.join(downloads_path,
                    webdata_export_file), destination_path)

    print("Files have been organized successfully.")


def main():
    organize_learning_path_graph_files()


if __name__ == '__main__':
    main()
