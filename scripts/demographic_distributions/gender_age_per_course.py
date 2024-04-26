import sqlite3
from dotenv import load_dotenv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from course_utilities import extract_course_year, calculate_age, identify_course
import os

load_dotenv()

MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')


def fetch_birth_year_gender_course_id() -> pd.DataFrame:
    conn = sqlite3.connect(MOOC_DB_LOCATION)
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


def plot_age_distribution(course_df: pd.DataFrame, course_name: str):
    male_population = course_df[course_df['Gender'] == 'm']
    female_population = course_df[course_df['Gender'] == 'f']

    male_and_female_population = course_df[
        (course_df['Gender'] == 'm') | (course_df['Gender'] == 'f')]
    standard_deviation = male_and_female_population['Age'].std()
    average = male_and_female_population['Age'].mean()
    median = male_and_female_population['Age'].median()

    male_population_standard_deviation = male_population['Age'].std()
    male_population_average = male_population['Age'].mean()
    male_population_median = male_population['Age'].median()

    female_population_standard_deviation = female_population['Age'].std()
    female_population_average = female_population['Age'].mean()
    female_population_median = female_population['Age'].median()

    print(f"Average age for all learners in {course_name} is {average}")
    print(f"""Standard deviation for all learners in {course_name} is {
        standard_deviation}""")
    print(f"Median age for all learners in {course_name} is {median}")
    print(f"""Average age for female learners in {
          course_name} is {female_population_average}""")
    print(f"""Standard deviation for female learners in {
          course_name} is {female_population_standard_deviation}""")
    print(f"""Median for female learners in {
          course_name} is {female_population_median}""")
    print(f"""Average age for male learners in {
          course_name} is {male_population_average}""")
    print(f"""Standard deviation for male learners in {
          course_name} is {male_population_standard_deviation}""")
    print(f"""Median for male learners in {
          course_name} is {male_population_median}""")

    # Uncomment if you want to include unknown and other
    # unknown_population = course_df[course_df['Gender'].isnull()]
    # other_population = course_df[course_df['Gender'] == 'o']

    independent_variable = "Age"

    sns.set_theme(style="whitegrid")
    colors = sns.color_palette(["#fe9929", "#1f78b4", "#f768a1", "#33a02c"])
    palette = sns.set_palette(colors)
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    plt.figure(figsize=(12, 8))
    sns.histplot(male_population, x=independent_variable,
                 binwidth=1, alpha=1.0, label='Male', palette=palette)
    sns.histplot(female_population, x=independent_variable,
                 binwidth=1, alpha=0.8, label='Female', palette=palette)

    # Uncomment if you want to include unknown and other
    # sns.histplot(unknown_population, x=independent_variable, binwidth=1,
    #              alpha=0.5, label='Prefer not to say / Unknown', palette=palette)
    # sns.histplot(other_population, x=independent_variable,
    #              binwidth=1, alpha=1.0, label='Other', palette=palette)

    plt.title(f"Age distribution per gender for {course_name}")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.legend(title='Gender', loc='upper right')
    plt.savefig(f"./figures/age/age_distribution_{course_name}.png")


def main() -> None:
    df = fetch_birth_year_gender_course_id()
    df['Course Year'] = df['Course ID'].apply(extract_course_year)
    df['Age'] = df.apply(lambda x: calculate_age(
        x['Course Year'], x['Birth year']), axis=1)
    # Exclude learners older than 99 years
    df = df[df['Age'] < 100]
    df['Course Name'] = df['Course ID'].apply(identify_course)
    courses = df['Course Name'].unique()
    for course in courses:
        course_df = df[df['Course Name'] == course]
        plot_age_distribution(course_df, course)


if __name__ == '__main__':
    main()
