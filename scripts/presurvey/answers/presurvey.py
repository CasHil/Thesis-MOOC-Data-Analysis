import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sqlite3
import json
import gzip
from dotenv import load_dotenv

load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')
COURSES = json.loads(os.getenv('COURSES'))
FIGURES_DIRECTORY = './figures'


def fetch_gender_and_course() -> pd.DataFrame:
    conn = sqlite3.connect(MOOC_DB_LOCATION)

    cur = conn.cursor()

    cur.execute("""
                SELECT UP.gender, E.course_id
                FROM user_profiles UP
                JOIN enrollments E ON UP.hash_id = E.hash_id
                """)

    data = cur.fetchall()
    df = pd.DataFrame(data, columns=["Gender", "Course ID"])

    cur.close()
    conn.close()

    return df


def find_all_presurveys_for_course_id(course_id: str) -> list[str]:
    return glob.glob(f"{WORKING_DIRECTORY}/{course_id}*/pre_survey*.txt", recursive=True)


def plot_answers(pre_survey: pd.DataFrame, question: list[str], survey_dir: str, question_text: str) -> None:

    order1 = ["Not at all important", "Slightly important",
              "Moderately important", "Important", "Very important"]
    order2 = ["Not at all important", "Slightly important",
              "Moderately important", "Very important", "Extremely important"]

    unique_values = pre_survey[question].unique()
    if "Important" in unique_values:
        order = order1
    else:
        order = order2

    plt.figure(figsize=(14, 10))
    sns.countplot(data=pre_survey, x=question, order=order)
    plt.title(question_text, fontsize=12)
    plt.xlabel('Answers', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95)
    plot_file_name = f"{survey_dir}/{question}.png"
    plt.savefig(plot_file_name)
    plt.close()

    # Extremely important
    # Very important
    # Moderately important
    # Slightly important
    # Not at all important

    # OR

    # Very important
    # Important
    # Moderately important
    # Slightly important
    # Not at all important


def find_all_presurveys_for_course_id(course_id: str) -> list[str]:
    return glob.glob(f"{WORKING_DIRECTORY}/{course_id}*/pre_survey*.txt", recursive=True)


def plot_answers(pre_survey: pd.DataFrame, question: str, survey_dir: str, question_text: str):
    order1 = ["Not at all important", "Slightly important",
              "Moderately important", "Important", "Very important"]
    order2 = ["Not at all important", "Slightly important",
              "Moderately important", "Very important", "Extremely important"]

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    palette = {
        "Male": default_colors[1],
        "Female": default_colors[0],
        "Prefer not to say": default_colors[2],
        "Other": default_colors[3]
    }

    unique_values = pre_survey[question].unique()
    if "Important" in unique_values:
        order = order1
    else:
        order = order2

    counts = pre_survey.groupby(
        ['Q4.4', question]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index().melt(
        id_vars='Q4.4', value_name='Percentage', var_name=question)

    plt.figure(figsize=(14, 10))
    sns.barplot(x=question, y='Percentage', hue="Q4.4",
                data=percentages, order=order, palette=palette)
    plt.title(question_text, fontsize=12)
    plt.xlabel('Answers', fontsize=10)
    plt.ylabel('Percentage', fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.legend(title='Gender')
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95)
    plot_file_name = f"{survey_dir}/{question}.png"
    plt.savefig(plot_file_name)
    plt.close()


def get_question_texts() -> dict[str, str]:
    return {
        "Q2.4_1": "How important were the following factors in your decision to enroll in this course? (Uniqueness of this course)",
        "Q2.4_2": "How important were the following factors in your decision to enroll in this course? (Potential usefulness of this course)",
        "Q2.4_3": "How important were the following factors in your decision to enroll in this course? (Interesting topic of this course)",
        "Q2.4_4": "How important were the following factors in your decision to enroll in this course? (Lecturer(s) involved with this course)",
        "Q2.4_5": "How important were the following factors in your decision to enroll in this course? (University(ies) involved with this course)",
        "Q2.4_6": "How important were the following factors in your decision to enroll in this course? (That the course is offered online)",
        "Q2.4_7": "How important were the following factors in your decision to enroll in this course? (The possibility to receive a certificate or credentials)",
        "Q108_1": "How important are for you the following elements in this course? (Videos)",
        "Q108_2": "How important are for you the following elements in this course? (Reading materials)",
        "Q108_3": "How important are for you the following elements in this course? (Forums)",
        "Q108_4": "How important are for you the following elements in this course? (Exercises, quizzes, assignments)",
        "Q3.2_5": "How important are for you the following elements in this course? (Group work, if applicable)",
        "Q3.2_6": "How important are for you the following elements in this course? (Feedback by instructors, if applicable)",
        "Q3.2_7": "How important are for you the following elements in this course? (Possibility to work on your own project, if applicable)"
    }


def gender_count_per_course(hash_ids: dict[str, set[str]]) -> None:
    conn = sqlite3.connect(MOOC_DB_LOCATION)

    for course in COURSES:
        cur = conn.cursor()
        hash_id_set = hash_ids[course]
        formatted_hash_id_set = ', '.join('?' for _ in hash_id_set)
        query = f"SELECT gender FROM user_profiles WHERE hash_id IN ({
            formatted_hash_id_set})"
        cur.execute(query, tuple(hash_id_set))
        genders = [item[0] for item in cur.fetchall()]
        print(f"Number of men who filled in presurvey for {
              course}: {genders.count('m')}")
        print(f"Number of women who filled in presurvey for {
              course}: {genders.count('f')}")
        print(f"Number of unknown who filled in presurvey for {
              course}: {genders.count(None)}")
        print(f"Number of other who filled in presurvey for {
              course}: {genders.count('o')}")


def extract_all_user_ids(log_files: list[str]) -> set[str]:
    user_ids = set()
    for log_file in log_files:
        print("Extracting user_ids from", log_file)
        with gzip.open(log_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    user_id = data.get("context", {}).get("user_id", "")
                    if user_id:
                        user_ids.add(user_id)
                except json.JSONDecodeError:
                    continue
    return user_ids


def extract_all_user_ids(log_files: list[str]) -> set[str]:
    user_ids = set()
    for log_file in log_files:
        print("Extracting user_ids from", log_file)
        with gzip.open(log_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    user_id = data.get("context", {}).get("user_id", "")
                    if user_id:
                        user_ids.add(user_id)
                except json.JSONDecodeError:
                    continue
    return user_ids


def find_log_files(base_path: str) -> list[str]:
    log_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".log.gz") and any(root.split(os.sep)[-1].startswith(course) for course in COURSES):
                log_files.append(os.path.join(root, file))
    return log_files


def main() -> None:
    with open('user_profiles.json', 'r', encoding='utf-8') as json_file:
        user_profiles = json.load(json_file)

    print("Finding log files...")
    log_files = find_log_files(WORKING_DIRECTORY)

    print("Extracting user ids from logs...")
    all_user_ids_from_logs = extract_all_user_ids(log_files)

    print("All user ids from logs", all_user_ids_from_logs
          )
    print("Filtering user profiles...")
    filtered_profiles = [
        profile for profile in user_profiles
        if profile['gender'] in ['m', 'f'] and profile['hash_id'] in all_user_ids_from_logs
    ]

    print("Filtered profiles", filtered_profiles)
    hash_ids_filtered = set(profile['hash_id']
                            for profile in filtered_profiles)
    # if not os.path.exists(FIGURES_DIR):
    #     os.makedirs(FIGURES_DIR)

    # QUESTION_TEXT = get_question_texts()

    hash_ids_per_course = {course: set() for course in COURSES}

    for course in COURSES:
        pre_surveys = find_all_presurveys_for_course_id(course)
        for ps in pre_surveys:
            pre_survey = pd.read_csv(ps, sep='\t')
            pre_survey = pre_survey.drop(0)
            # path_parts = ps.split("\\")
            # course_id = path_parts[-2].split("_")[0]
            # year_and_course_run = "_".join(path_parts[-2].split("_")[1:])

            # survey_dir = f'{FIGURES_DIR}/{course_id}/{year_and_course_run}'
            # if not os.path.exists(survey_dir):
            #     os.makedirs(survey_dir)

            # for question in QUESTION_TEXT.keys():
            #     if question in pre_survey.columns:
            #         plot_answers(pre_survey, question, survey_dir, QUESTION_TEXT[question])

            hash_ids = pre_survey['hash_id'].unique()
            for hash_id in hash_ids:
                hash_ids_per_course[course].add(hash_id)

    # gender_count_per_course(hash_ids_per_course)

    for course in COURSES:
        print(f"Number of hash_ids for {course}: {
              len(hash_ids_per_course[course])}")
        print(f"Hash ids filtered {hash_ids_filtered}")
        print(f"Hash ids per course {hash_ids_per_course[course]}")
        print(hash_ids_per_course[course].intersection(hash_ids_filtered))
        intersect = hash_ids_per_course[course].intersection(hash_ids_filtered)
        men = [profile for profile in filtered_profiles if profile['hash_id']
               in intersect and profile['gender'] == 'm']

        women = [profile for profile in filtered_profiles if profile['hash_id']
                 in intersect and profile['gender'] == 'f']

        print(
            [profile for profile in filtered_profiles if profile['hash_id'] in intersect])
        print(len(men))
        print(len(women))
        # Number of hash_ids for EX101x: 9666
# 2095
# 991
# Number of hash_ids for ST1x: 3846
# 1222
# 601
# Number of hash_ids for UnixTx: 1016
# 445
# 94
# Number of hash_ids for FP101x: 3487
# 2771
# 192
        # Save the intersection of hash_ids_filtered and hash_ids_per_course[course] to a json file
        with open(f'{course}_hash_ids.json', 'w') as json_file:
            json.dump(list(hash_ids_per_course[course].intersection(
                hash_ids_filtered)), json_file)


if __name__ == "__main__":
    main()
