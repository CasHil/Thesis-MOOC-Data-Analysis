import sqlite3
from dotenv import load_dotenv
import pandas as pd
from course_utilities import identify_course
import os

load_dotenv()

COURSES = ['EX101x', 'ST1x', 'UnixTx']

MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')

def fetch_gender_and_course():
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

def gender_count_per_course(course_df, course_id):
    male_population = course_df[course_df['Gender'] == 'm']
    female_population = course_df[course_df['Gender'] == 'f']
    unknown_population = course_df[course_df['Gender'].isnull()]
    other_population = course_df[course_df['Gender'] == 'o']

    male_count = len(male_population)
    female_count = len(female_population)
    unknown_count = len(unknown_population)
    other_count = len(other_population)

    print(f"Number of men for course {course_id}: {male_count}")
    print(f"Number of women for course {course_id}: {female_count}")
    print(f"Number of unknown for course {course_id}: {unknown_count}")
    print(f"Number of other for course {course_id}: {other_count}")

def main():
    df = fetch_gender_and_course()
    df['Course Name'] = df['Course ID'].apply(identify_course)

    for COURSE in COURSES:
        gender_count_per_course(df[df["Course Name"] == COURSE], COURSE)

if __name__ == '__main__':
    main()