import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob 
import sqlite3

WORKING_DIR = 'W:/staff-umbrella/gdicsmoocs/Working copy'
COURSES = ['EX101x', 'ST1x', 'UnixTx']
FIGURES_DIR = './figures'


def fetch_gender_and_course():
    conn = sqlite3.connect("W:/staff-umbrella/gdicsmoocs/Working copy/scripts/thesis_db")

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

def identify_course(course_id):
    if 'EX101x' in course_id:
        return 'EX101x'
    elif 'ST1x' in course_id:
        return 'ST1x'
    elif 'UnixTx' in course_id:
        return 'UnixTx'
    else:
        return 'Other'
    
def find_all_presurveys_for_course_id(course_id):
    return glob.glob(f"{WORKING_DIR}/{course_id}*/pre_survey*.txt", recursive=True)

def plot_answers(pre_survey, question, survey_dir, question_text):
  
    order1 = ["Not at all important", "Slightly important", "Moderately important", "Important", "Very important"]
    order2 = ["Not at all important", "Slightly important", "Moderately important", "Very important", "Extremely important"]

  
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

def find_all_presurveys_for_course_id(course_id):
    return glob.glob(f"{WORKING_DIR}/{course_id}*/pre_survey*.txt", recursive=True)

def plot_answers(pre_survey, question, survey_dir, question_text):
    order1 = ["Not at all important", "Slightly important", "Moderately important", "Important", "Very important"]
    order2 = ["Not at all important", "Slightly important", "Moderately important", "Very important", "Extremely important"]

  
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
    
    counts = pre_survey.groupby(['Q4.4', question]).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    percentages = percentages.reset_index().melt(id_vars='Q4.4', value_name='Percentage', var_name=question)


    plt.figure(figsize=(14, 10))
    sns.barplot(x=question, y='Percentage', hue="Q4.4", data=percentages, order=order, palette=palette)
    plt.title(question_text, fontsize=12)
    plt.xlabel('Answers', fontsize=10)
    plt.ylabel('Percentage', fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.legend(title='Gender')
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.05, right=0.95)
    plot_file_name = f"{survey_dir}/{question}.png"
    plt.savefig(plot_file_name)
    plt.close()

def get_question_texts():
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

def gender_count_per_course(hash_ids):
    conn = sqlite3.connect("W:/staff-umbrella/gdicsmoocs/Working copy/scripts/thesis_db")

    for course in COURSES:
        cur = conn.cursor()
        hash_id_set = hash_ids[course]
        formatted_hash_id_set = ', '.join('?' for _ in hash_id_set)
        query = f"SELECT gender FROM user_profiles WHERE hash_id IN ({formatted_hash_id_set})"
        cur.execute(query, tuple(hash_id_set))
        genders = [item[0] for item in cur.fetchall()]
        print(f"Number of men who filled in presurvey for {course}: {genders.count('m')}")
        print(f"Number of women who filled in presurvey for {course}: {genders.count('f')}")
        print(f"Number of unknown who filled in presurvey for {course}: {genders.count(None)}")
        print(f"Number of other who filled in presurvey for {course}: {genders.count('o')}")

def main():
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)

    QUESTION_TEXT = get_question_texts()

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
    
    gender_count_per_course(hash_ids_per_course)
if __name__ == "__main__":
    main()