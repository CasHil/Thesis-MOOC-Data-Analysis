from pymongo import MongoClient
import os

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

collection = db.quiz_sessions

print(courses_with_due_dates)

for course_folder in courses_with_due_dates:
    quiz_query = {"course_id": course_folder}
    result = collection.count_documents(quiz_query)
    print(result)
    if collection.count_documents(quiz_query) > 0:
        print(f"Course {course_folder} has quiz sessions and due dates.")