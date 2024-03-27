import sqlite3
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
from course_utilities import identify_course
import os

load_dotenv()

MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')

def fetch_education_level_and_demographic_info() -> pd.DataFrame:
    conn = sqlite3.connect(MOOC_DB_LOCATION)
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

def plot_education_distribution(course_df: pd.DataFrame, course_name: str) -> None:
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

def expand_education_level(education_df: pd.DataFrame) -> None:
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
    education_df['Level of Education'] = education_df['Level of Education'].map(education_map)
    education_df = education_df.dropna()
    return education_df

def main() -> None:
    education_df = fetch_education_level_and_demographic_info()
    education_df = expand_education_level(education_df)
    education_df['Course Name'] = education_df['Course ID'].apply(identify_course)
    courses = education_df['Course Name'].unique()
    for course in courses:
        course_df = education_df[education_df['Course Name'] == course]
        plot_education_distribution(course_df, course)

if __name__ == '__main__':
    main()