import sqlite3
from dotenv import load_dotenv
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

def fetch_birth_year_gender_course_id():
    conn = sqlite3.connect("W:/staff-umbrella/gdicsmoocs/Working copy/scripts/thesis_db")
    cur = conn.cursor()

    cur.execute("""
                SELECT UP.level_of_education, UP.gender, E.course_id
                FROM user_profiles UP
                JOIN enrollments E ON UP.hash_id = E.hash_id
                """)

    data = cur.fetchall()
    df = pd.DataFrame(data, columns=["Level of Education", "Gender", "Course ID"])

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

def plot_education_distribution(course_df, course_name):
    df_copy = course_df.copy()
    df_copy['GenderLabel'] = df_copy['Gender'].map({'m': 'Male', 'f': 'Female', 'o': 'Other'}).fillna('Prefer not to say / Unknown')
    categories_order = ['Male', 'Female', 'Other', 'Prefer not to say / Unknown']
    
    education_gender_counts = df_copy.groupby(['Level of Education', 'GenderLabel']).size().unstack(fill_value=0)
    education_gender_counts = education_gender_counts.reindex(columns=categories_order, fill_value=0)
    
    education_gender_counts['Total'] = education_gender_counts.sum(axis=1)
    education_gender_counts = education_gender_counts.sort_values(by='Total', ascending=False).drop(columns=['Total'])

    if not education_gender_counts.empty:
        ax = education_gender_counts.plot(kind='bar', stacked=True, figsize=(12, 8))
        plt.title('Education Distribution per Gender in ' + course_name)
        plt.xlabel('Education Level')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.legend(title='Gender')
        
        for container in ax.containers:
            plt.setp(container, width=0.85)
        
        plt.savefig(f"./figures/education/education_distribution_{course_name}.png")
    else:
        print(f"No valid data in 'GenderLabel' column for {course_name}")

def expand_education_level(df):
    education_map = {
        "none": "No formal education",
        "a": "College",
        "hs": "Secondary school",
        "m": "Master's",
        "jhs": "Secondary school (incomplete)",
        "p_oth": "Professional",
        "p_se": "(Post-)graduate",
        "b": "Bachelor's",
        "other": "Other educational background"
    }
    df['Level of Education'] = df['Level of Education'].map(education_map)
    df = df.dropna()
    return df

def main():
    df = fetch_birth_year_gender_course_id()
    df = expand_education_level(df)
    df['Course Name'] = df['Course ID'].apply(identify_course)
    courses = df['Course Name'].unique()
    for course in courses:
        course_df = df[df['Course Name'] == course]
        plot_education_distribution(course_df, course)

if __name__ == '__main__':
    main()