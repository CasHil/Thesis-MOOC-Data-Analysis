# Find all folders starting with "EX101x", "ST1x", "FP101x" and "UnixTx". Go into this folder, find the file name like this: DelftX-EX101x-3T2015-course_structure-prod-analytics.json
# Then keep all the course folder names that have a due date in any of the lines, some line with "due" in it and not "due": null

# Path: db/find_courses_with_due_dates_and_quizzes.py

from pymongo import MongoClient
from tqdm import tqdm
import os
import json

client = MongoClient('mongodb://localhost:27017/')
db = client["edx_test"]
collection = db.quiz_sessions

def get_course_folders():
    course_folders = []
    for folder in os.listdir("W:\staff-umbrella\gdicsmoocs\Working copy"):
        if folder.startswith("EX101x") or folder.startswith("ST1x") or folder.startswith("FP101x") or folder.startswith("UnixTx"):
            course_folders.append(folder)
    return course_folders


def get_course_structure(course_folder):
    course_folder_without_run = "-".join(course_folder.split("_")[:2]).replace("[", "").replace("]", "")
    course_structure_file = f"W:\staff-umbrella\gdicsmoocs\Working copy\{course_folder}\DelftX-{course_folder_without_run}-course_structure-prod-analytics.json"
    with open(course_structure_file, "r") as file:
        return file.readlines()


def has_due_date(course_structure):
    count = 0
    for line in course_structure:
        # print(line)
        if "due" in line and "null" not in line:
            count += 1
    return count > 5

courses_with_due_dates = []
course_folders = get_course_folders()
for course_folder in course_folders:
    course_structure = get_course_structure(course_folder)
    if has_due_date(course_structure):
        course_folder_without_run = "course-v1:DelftX+" + "+".join(course_folder.split("_")[:2]).replace("[", "").replace("]", "")
        courses_with_due_dates.append(course_folder_without_run)    
    else:
        print(f"{course_folder} has no due date.")

client = MongoClient("mongodb://localhost:27017/")
print("Connected to MongoDB")
db = client["edx_test"]
print("Connected to database")

print(courses_with_due_dates)

for course_folder in courses_with_due_dates:
    quiz_query = {"course_id": course_folder}
    result = collection.count_documents(quiz_query)
    print(result)
    if collection.count_documents(quiz_query) > 0:
        print(f"Course {course_folder} has quiz sessions and due dates.")