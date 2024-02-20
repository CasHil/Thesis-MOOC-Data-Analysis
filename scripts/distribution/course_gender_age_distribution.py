import psycopg2
from dotenv import load_dotenv
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

def fetch_birth_year_gender_course_id():
    conn = psycopg2.connect(
        host="localhost",
        database="thesis",
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

    cur = conn.cursor()

    cur.execute("""
                SELECT UP.year_of_birth, UP.gender, E.course_id
                FROM user_profiles UP
                JOIN enrollments E ON UP.hash_id = E.hash_id
                """)

    data = cur.fetchall()
    df = pd.DataFrame(data, columns=["Birth year", "Gender", "Course ID"])

    cur.close()
    conn.close()

    return df

def extract_course_year(course_id):
    return int(course_id[-4:])

def calculate_age(course_year, birth_year):
    return course_year - birth_year

def identify_course(course_id):
    if 'EX101x' in course_id:
        return 'EX101x'
    elif 'ST1x' in course_id:
        return 'ST1x'
    elif 'UnixTx' in course_id:
        return 'UnixTx'
    else:
        return 'Other'

def plot_age_distribution(course_df, course_name):
    male_population = course_df[course_df['Gender'] == 'm']
    female_population = course_df[course_df['Gender'] == 'f']
    unknown_population = course_df[course_df['Gender'].isnull()]
    other_population = course_df[course_df['Gender'] == 'o']

    independent_variable = "Age"
    
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette(["#fe9929", "#1f78b4", "#f768a1", "#33a02c"])
    palette = sns.set_palette(colors)
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    plt.figure(figsize=(12, 8))
    sns.histplot(male_population, x=independent_variable, binwidth=1, alpha=1.0, label='Male', palette=palette)
    sns.histplot(female_population, x=independent_variable, binwidth=1, alpha=0.8, label='Female', palette=palette)
    sns.histplot(unknown_population, x=independent_variable, binwidth=1, alpha=0.5, label='Prefer not to say / Unknown', palette=palette)
    sns.histplot(other_population, x=independent_variable, binwidth=1, alpha=1.0, label='Other', palette=palette)

    plt.title(f"Age distribution per gender for {course_name}")    
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.legend(title='Gender', loc='upper right')
    plt.savefig(f"./figures/age_distribution_{course_name}.png")

def main():
    df = fetch_birth_year_gender_course_id()
    df['Course Year'] = df['Course ID'].apply(extract_course_year)
    df['Age'] = df.apply(lambda x: calculate_age(x['Course Year'], x['Birth year']), axis=1)
    # Exclude learners older than 99 years
    df = df[df['Age'] < 100]
    df['Course Name'] = df['Course ID'].apply(identify_course)
    courses = df['Course Name'].unique()
    for course in courses:
        course_df = df[df['Course Name'] == course]
        plot_age_distribution(course_df, course)

if __name__ == '__main__':
    main()

# # Extract the course year from the last four characters of the course_id
# df['Course Year'] = df['Course ID'].apply(lambda x: int(x[-4:]))

# # Calculate age at time of course
# df['Age'] = df['Course Year'] - df['Birth year']


# df['Course Name'] = df['Course ID'].apply(identify_course)

# # Plotting
# sns.set_theme(style="whitegrid")
# colors = sns.color_palette(["#fe9929", "#1f78b4", "#f768a1", "#33a02c"])
# palette = sns.set_palette(colors)
# sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


# courses = df['Course Name'].unique()

# for course in courses:
#     plt.figure(figsize=(12, 8))
#     course_df = df[df['Course Name'] == course]
    
#     male_population = course_df[course_df['Gender'] == 'm']
#     female_population = course_df[course_df['Gender'] == 'f']
#     unknown_population = course_df[course_df['Gender'].isnull()]
#     other_population = course_df[course_df['Gender'] == 'o']

#     sns.histplot(male_population, x="Age", binwidth=1, alpha=1.0, label='Male', palette=palette)
#     sns.histplot(female_population, x="Age", binwidth=1, alpha=0.8, label='Female', palette=palette)
#     sns.histplot(unknown_population, x="Age", binwidth=1, alpha=0.5, label='Prefer not to say / Unknown', palette=palette)
#     sns.histplot(other_population, x="Age", binwidth=1, alpha=1.0, label='Other', palette=palette)

#     plt.title(f"Age distribution per gender for {course}")
    
#     plt.xlabel("Age")
#     plt.ylabel("Count")
#     plt.tight_layout()
#     plt.legend(title='Gender')
#     plt.savefig(f"./figures/age_distribution_{course}.png")