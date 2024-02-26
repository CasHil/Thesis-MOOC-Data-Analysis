import sqlite3
from dotenv import load_dotenv
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

load_dotenv()

def fetch_birth_year_and_gender():
    conn = sqlite3.connect("W:/staff-umbrella/gdicsmoocs/Working copy/scripts/thesis_db")

    cur = conn.cursor()

    cur.execute("""
                SELECT year_of_birth, gender FROM user_profiles
                """)

    data = cur.fetchall()
    df = pd.DataFrame(data, columns=["Birth year", "Gender"])
    
    cur.close()
    conn.close()

    return df

def plot_birth_year_distribution_for_all_courses(course_df):
    male_population = course_df[course_df['Gender'] == 'm']
    female_population = course_df[course_df['Gender'] == 'f']
    unknown_population = course_df[course_df['Gender'].isnull()]
    other_population = course_df[course_df['Gender'] == 'o']
    
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette(["#fe9929", "#1f78b4", "#f768a1", "#33a02c"])
    palette = sns.set_palette(colors)
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    plt.figure(figsize=(12, 8))
    sns.histplot(male_population, x="Birth year", binwidth=1, alpha=1.0, label='Male', palette=palette)
    sns.histplot(female_population, x="Birth year", binwidth=1, alpha=0.8, label='Female', palette=palette)
    sns.histplot(unknown_population, x="Birth year", binwidth=1, alpha=0.5, label='Prefer not to say / Unknown', palette=palette)
    sns.histplot(other_population, x="Birth year", binwidth=1, alpha=1.0, label='Other', palette=palette)

    plt.title("Distribution of birth years")
    plt.legend(title='Gender', loc='upper right')
    plt.savefig("./figures/age/distribution_of_birth_years.png")

def main():
    df = fetch_birth_year_and_gender()
    plot_birth_year_distribution_for_all_courses(df)

if __name__ == '__main__':
    main()