import sqlite3
import glob 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

WORKING_DIR = 'W:/staff-umbrella/gdicsmoocs/Working copy'
DB_LOCATION = WORKING_DIR + '/scripts/thesis_db'
COURSES = ['EX101x', 'ST1x', 'UnixTx']

def find_all_log_files_for_course_id(course_id):
    return glob.glob(f"{WORKING_DIR}/{course_id}*/*.log", recursive=True)

# def find_all_pause_events_for_course_id(course_id):
#     pause_events = []
#     for log_file in find_all_log_files_for_course_id(course_id):
#         with open(log_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if 'pause_video' in line:
#                     pause_events.append(line)
#     return pause_events

# Temporarily only use the first 5 log files for a course for testing.
def find_all_pause_events_for_course_id(course_id):
    pause_events = []
    for i, log_file in enumerate(find_all_log_files_for_course_id(course_id)):
        if i >= 5:  # Only process the first 5 log files
            break
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'pause_video' in line:
                    pause_events.append(line)
    return pause_events

def calculate_average_pause_count_per_gender(course_id):
    con = sqlite3.connect(DB_LOCATION)
    cur = con.cursor()
    pause_events = find_all_pause_events_for_course_id(course_id)

    user_pause_counts = {}
    for pause_event in pause_events:
        if '"user_id":null' in pause_event:
            continue
        user_id = pause_event.split('"user_id":"')[1].split('"')[0]
        if user_id in user_pause_counts:
            user_pause_counts[user_id] += 1
        else:
            user_pause_counts[user_id] = 1

    gender_pause_counts = {'m': [], 'f': [], 'o': [], None: []}
    for user_id, pause_count in user_pause_counts.items():
        query = 'SELECT gender FROM user_profiles WHERE hash_id = ?'
        cur.execute(query, (user_id,))
        result = cur.fetchone()
        if result:
            gender = result[0]
            gender_pause_counts[gender].append(pause_count)

    gender_pause_averages = {}
    for gender, pause_counts in gender_pause_counts.items():
        if pause_counts:
            average = sum(pause_counts) / len(pause_counts)
            gender_pause_averages[gender] = average
        
    cur.close()
    con.close()

    return gender_pause_averages

def graph_average_pauses_per_gender(average_pauses_per_course):
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    rows = []
    for course, genders in average_pauses_per_course.items():
        for gender, value in genders.items():
            rows.append({"Course": course, "Gender": gender, "Value": value})

    df = pd.DataFrame(rows)
    gender_map = {'m': 'Male', 'f': 'Female', 'o': 'Other', None: 'Prefer not to say / Unknown'}
    df['Gender'] = df['Gender'].map(gender_map)

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette(["#fe9929", "#1f78b4", "#f768a1", "#33a02c"])
    sns.set_palette(colors)
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    g = sns.catplot(x='Course', y='Value', hue='Gender', kind='bar', data=df, height=6, aspect=1.5)
    plt.xticks(rotation=45)
    plt.ylabel('Average Pauses')
    plt.xlabel('Course')
    plt.title('Average Pauses by Gender across Courses')

    plt.margins(0.1)

    g.savefig('graphs/average_pauses_per_gender.png')

def main():
    average_pauses_per_course = {}
    for course_id in COURSES:
        print(f"Analysing pauses in course: {course_id}")
        gender_pause_averages = calculate_average_pause_count_per_gender(course_id)
        print(f"Average pauses in course {course_id}: {gender_pause_averages}") 
        average_pauses_per_course[course_id] = gender_pause_averages
    
    graph_average_pauses_per_gender(average_pauses_per_course)

if __name__ == '__main__':
    main()