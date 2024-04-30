# In this script, we will find the mandatory elements per assessment period. This will be done by looking at users who have a perfect score for the MOOC.
# The reason for doing it this way is that, when using the course elements due from the metadata, not a single user has all the mandatory elements completed on time.

import pandas as pd
import numpy as np
import os
import json
from pymongo import MongoClient
from pymongo.database import Database
from datetime import datetime
import re


def get_courses(db: Database) -> pd.DataFrame:
    courses = db["courses"].find()

    courses_df = pd.DataFrame(courses)

    if courses_df.empty:
        raise ValueError("No courses found.")

    courses_df['start_time'] = pd.to_datetime(courses_df['start_time'])
    courses_df['end_time'] = pd.to_datetime(courses_df['end_time'])

    return courses_df

def get_mandatory_quizzes(db: Database, course_id: str, learner_id: str) -> set[str]:
    print("Getting mandatory quizzes for learner", learner_id, "in course", course_id)
    quiz_sessions = db["quiz_sessions"].find({"course_learner_id": learner_id})
    quiz_sessions_df = pd.DataFrame(quiz_sessions)

    if quiz_sessions_df.empty:
        return set()

    return set(quiz_sessions_df["block_id"].unique())

def get_mandatory_ora_sessions(db: Database, course_id: str, learner_id: str) -> set[str]:
    print("Getting mandatory ora sessions for learner", learner_id, "in course", course_id)
    ora_sessions = db["ora_sessions"].find({"course_learner_id": learner_id})
    ora_sessions_df = pd.DataFrame(ora_sessions)

    if ora_sessions_df.empty:
        return set()

    return set(ora_sessions_df["block_id"].unique())

def get_mandatory_problems(db: Database, course_id: str, learner_id: str) -> set[str]:
    print("Getting mandatory problems for learner", learner_id, "in course", course_id)
    problems = db["submissions"].find({"course_learner_id": learner_id})
    problems_df = pd.DataFrame(problems)

    if problems_df.empty:
        return set()
    
    return set(problems_df["question_id"].unique())

def find_mandatory_elements_by_best_learner(course_id: str, perfect_learner: str, db: Database) -> set[str]:
    course = db["courses"].find_one({"course_id": course_id})
    if course is None:
        raise ValueError(f"Course {course_id} not found.")
    
    mandatory_quizzes = get_mandatory_quizzes(db, course_id, perfect_learner)
    mandatory_ora_sessions = get_mandatory_ora_sessions(db, course_id, perfect_learner)
    mandatory_problems = get_mandatory_problems(db, course_id, perfect_learner)

    return mandatory_quizzes.union(mandatory_ora_sessions).union(mandatory_problems)
    
def group_mandatory_elements_by_deadline(course_id: str, mandatory_elements: set[str], db: Database) -> pd.Series:
    metadata = db["metadata"].find_one({"course_id": course_id})["object"]
    
    due_dates: dict = metadata["element_time_map_due"]
    element_time_map: dict = metadata["element_time_map"]
    child_parent_map: dict = metadata["child_parent_map"]

    deadlines_with_mandatory_elements = set(pd.Timestamp(due_date) for due_date in due_dates.values())
    deadlines_with_mandatory_elements = {deadline: [] for deadline in deadlines_with_mandatory_elements}

    mandatory_elements.discard(None)
    for element in mandatory_elements:
        print(element)
        if due_dates.get(element) is not None:
            print(element, "1")
            due_date = due_dates.get(element)
        elif element_time_map.get(element) is not None:
            print(element, "2")
            due_date = element_time_map.get(element)
        elif child_parent_map.get(element) is not None:
            print(element, "3")
            parent = child_parent_map.get(element)
            while parent not in due_dates and "chapter" not in parent:
                parent = child_parent_map[parent]
            due_date = due_dates.get(parent)
            
        if due_date is not None:
            # Append the element to the first timestamp that comes after due_date
            for key in sorted(deadlines_with_mandatory_elements.keys()):
                if key > pd.Timestamp(due_date):
                    deadlines_with_mandatory_elements[key].append(element)
                    break           

    
    string_deadlines_with_mandatory_elements = {str(deadline): elements for deadline, elements in deadlines_with_mandatory_elements.items()}

    return string_deadlines_with_mandatory_elements
   
def find_best_learner(course_id: str, db: Database) -> str:
    course_id = course_id.replace("_", "")
    print("Finding best learner for course", course_id)
    pipeline = [
        {"$match": {"course_id": course_id}},  # Match documents for the specified course
        {"$sort": {"final_grade": -1}},         # Sort by final_grade in descending order
        {"$limit": 1},                         # Limit to get only the top result
        {"$project": {"course_learner_id": 1}} # Project only the course_learner_id field
    ]
    result = db['course_learner'].aggregate(pipeline)
    result = list(result)
    if len(result) == 0:
        return ""
    best_learner = result[0]["course_learner_id"]
    return best_learner
    
    
def main():
    client = MongoClient("mongodb://localhost:27017/")
    print("Connected to MongoDB")
    db = client["edx_test"]
    courses = get_courses(db)
    data = {}

    for course_id in courses["course_id"]:
        best_learner = find_best_learner(course_id, db)
        if best_learner == "":
            continue
        mandatory_elements = find_mandatory_elements_by_best_learner(course_id, best_learner, db)
        mandatory_elements_by_deadline = group_mandatory_elements_by_deadline(course_id, mandatory_elements, db)

        data[course_id] = mandatory_elements_by_deadline

    with open('mandatory_elements_per_course.json', 'w') as f:
        # For 
        json.dump(data, f)


if __name__ == "__main__":
    main()