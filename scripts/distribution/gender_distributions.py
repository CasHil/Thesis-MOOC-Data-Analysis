import psycopg2
from dotenv import load_dotenv
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

load_dotenv()

conn = psycopg2.connect(
    host="localhost",
    database="thesis",
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

cur = conn.cursor()

cur.execute("""
            SELECT year_of_birth FROM user_profiles WHERE gender = 'm'
            """)

male_population = cur.fetchall()
male_population = pd.DataFrame(male_population, columns=["Birth year"])

cur.execute("""
            SELECT year_of_birth FROM user_profiles WHERE gender = 'f'
            """)    

female_population = cur.fetchall()
female_population = pd.DataFrame(female_population, columns=["Birth year"])

cur.execute("""
            SELECT year_of_birth FROM user_profiles WHERE
            gender = 'o'
            """)

other_population = cur.fetchall()
other_population = pd.DataFrame(other_population, columns=["Birth year"])


cur.execute("""
            SELECT year_of_birth FROM user_profiles WHERE gender IS NULL
            """)
unknown_population = cur.fetchall()
unknown_population = pd.DataFrame(unknown_population, columns=["Birth year"])

cur.close()
conn.close()

# Plotting the data
sns.set_theme(style="whitegrid")
colors = sns.color_palette(["#fe9929", "#1f78b4", "#f768a1", "#33a02c"])
palette = sns.set_palette(colors)
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
male_hist = sns.histplot(male_population, x="Birth year", binwidth=1, alpha=0.8, label='Male',palette=palette)
sns.histplot(female_population, x="Birth year", binwidth=1, alpha=0.8, label='Female', palette=palette)
sns.histplot(unknown_population, x="Birth year", binwidth=1, alpha=0.9, label='Prefer not to say / Unknown', palette=palette)
sns.histplot(other_population, x="Birth year", binwidth=1, alpha=1.0, label='Other', palette=palette)
plt.title("Distribution of birth years")

plt.legend(title='Gender')
plt.show()