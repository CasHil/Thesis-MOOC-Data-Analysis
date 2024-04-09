import os
import sqlite3
from dotenv import load_dotenv
import json
from datetime import datetime
import re
load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
SCRIPTS_DIRECTORY = os.getenv('SCRIPTS_DIRECTORY')
MOOC_DB = os.getenv('MOOC_DB_LOCATION')
COURSES = ['EX101x', 'FP101x', 'ST1x', 'UnixTx']


def check_valid_course(course_id: str) -> str:
    if 'EX101x' in course_id or 'ST1x' in course_id or 'UnixTx' in course_id or 'FP101x' in course_id:
        return True
    return False


def find_course_directories_and_metadata_files() -> dict[str, list[str]]:
    metadata_files_per_course_run = {}

    required_metadata_files = [
        "course_structure-prod-analytics.json",
    ]

    for dir_name in os.listdir(WORKING_DIRECTORY):
        if check_valid_course(dir_name):
            print(dir_name)
            course_path = os.path.join(
                os.path.abspath(WORKING_DIRECTORY), dir_name)
            files = []
            for file_name in os.listdir(course_path):
                file_name: str
                if any(file_name.endswith(specific_file_name) for specific_file_name in required_metadata_files):
                    files.append(os.path.join(course_path, file_name))

            if len(files) != len(required_metadata_files):
                missing_files = []
                for required_metadata_file in required_metadata_files:
                    if not required_metadata_file in files:
                        missing_files.append(required_metadata_file)
                raise ValueError("Missing metadata files:", missing_files)

            metadata_files_per_course_run[dir_name] = files
    return metadata_files_per_course_run


def extract_course_information(files):
    course_metadata_map = {}
    print(files)

    if not any("course_structure" in file for file in files):
        print("Course structure file is missing!")
        return course_metadata_map

    course_structure_file = open(next(
        file for file in files if "course_structure" in file))

    child_parent_map = {}
    element_time_map = {}
    element_time_map_due = {}
    element_type_map = {}
    element_without_time = []
    quiz_question_map = {}
    block_type_map = {}
    order_map = {}
    element_name_map = {}

    json_object = json.loads(course_structure_file.read())
    for record in json_object:
        if json_object[record]["category"] == "course":
            course_id = record.replace("block-", "course-").replace(
                "+type@course+block@course", "").replace("i4x://", "").replace("course/", "")
            course_metadata_map["course_id"] = course_id
            course_metadata_map["course_name"] = json_object[record]["metadata"]["display_name"]
            start_date = json_object[record]["metadata"]["start"]
            end_date = json_object[record]["metadata"]["end"]
            course_metadata_map["start_time"] = start_date
            course_metadata_map["end_time"] = end_date
            element_position = 0
            for child in json_object[record]["children"]:
                element_position += 1
                child_parent_map[child] = record
                order_map[child] = element_position
            element_time_map[record] = start_date
            element_type_map[record] = json_object[record]["category"]
        else:
            element_id = record
            element_name_map[element_id] = json_object[element_id]["metadata"].get(
                "display_name", "")
            element_position = 0
            for child in json_object[element_id].get("children", []):
                element_position += 1
                child_parent_map[child] = element_id
                order_map[child] = element_position
            if "start" in json_object[element_id]["metadata"]:
                element_time_map[element_id] = json_object[element_id]["metadata"]["start"]
            else:
                element_without_time.append(element_id)
            if "due" in json_object[element_id]["metadata"]:
                element_time_map_due[element_id] = json_object[element_id]["metadata"]["due"]
            element_type_map[element_id] = json_object[element_id]["category"]
            if json_object[element_id]["category"] == "problem":
                quiz_question_map[element_id] = json_object[element_id]["metadata"].get(
                    "weight", 1.0)
            if json_object[element_id]["category"] == "sequential":
                block_type_map[element_id] = json_object[element_id]["metadata"].get(
                    "display_name", "")

    for element_id in element_without_time:
        element_start_time = element_time_map.get(
            child_parent_map[element_id], "")
        element_time_map[element_id] = element_start_time

    course_metadata_map["element_time_map"] = element_time_map
    course_metadata_map["element_time_map_due"] = element_time_map_due
    course_metadata_map["element_type_map"] = element_type_map
    course_metadata_map["quiz_question_map"] = quiz_question_map
    course_metadata_map["child_parent_map"] = child_parent_map
    course_metadata_map["block_type_map"] = block_type_map
    course_metadata_map["order_map"] = order_map
    course_metadata_map["element_name_map"] = element_name_map
    print("Metadata map ready")
    return course_metadata_map


def sql_insert(table: str, data_rows: list[dict[str, str]], connection: sqlite3.Connection):
    cursor = connection.cursor()

    if table != "forum_interaction":
        cursor.execute(f"DELETE FROM {table}")

    placeholders = ", ".join(["?"] * len(data_rows[0]))
    columns = ", ".join(data_rows[0].keys())
    query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

    prepared_data_rows = []
    for v in data_rows:
        row = []
        for field, value in v.items():
            if "time" in field and value is not None:
                row.append(datetime.fromisoformat(value))
            else:
                row.append(value)
        prepared_data_rows.append(tuple(row))

    cursor.executemany(query, prepared_data_rows)
    connection.commit()
    rows_added = cursor.rowcount
    if rows_added > 0 and table != "forum_interaction":
        print(f"Successfully added {len(data_rows)} to {table}")

    cursor.close()
    connection.close()


def main() -> None:
    course_directories_and_metadata_files = find_course_directories_and_metadata_files()
    connection = sqlite3.connect(MOOC_DB)
    course_runs_metadata = []
    for course_run, course_run_metadata_files in course_directories_and_metadata_files.items():
        course_metadata_map = extract_course_information(
            course_run_metadata_files)
        metadata_map_json_string = json.dumps(course_metadata_map)
        course_runs_metadata.append(
            {"course_id": course_run, "object": metadata_map_json_string})

    sql_insert('metadata', course_runs_metadata, connection)


if __name__ == '__main__':
    main()
