import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

import json
import os
from dotenv import load_dotenv
import pandas as pd
import glob
from dotenv import load_dotenv
from sklearn.metrics import silhouette_score
import numpy as np
import sqlite3

load_dotenv()

COURSES = json.loads(os.getenv("COURSES"))
WORKING_DIRECTORY = os.getenv("WORKING_DIRECTORY")
MOOC_DB_LOCATION = os.getenv("MOOC_DB_LOCATION")

for dependency in ("punkt", "stopwords"):
    nltk.download(dependency)


def find_genders_by_user_ids(user_ids: list) -> dict:
    con = sqlite3.connect(MOOC_DB_LOCATION)
    cur = con.cursor()
    query = 'SELECT hash_id, gender FROM user_profiles WHERE hash_id IN ({})'.format(
        ','.join('?'*len(user_ids)))
    cur.execute(query, user_ids)
    results = cur.fetchall()
    return {result[0]: result[1] for result in results}


class SurveyQuestions():
    def __init__(self):
        self.survey_data = {
            '2015T3': {'closed': 'Q3.5', 'open': 'Q3.5_0_TEXT'},
            '2016T3': {'closed': 'Q2.5', 'open': 'Q2.5_6_TEXT'},
            '2016T3a': {'closed': 'Q2.5', 'open': 'Q2.5_6_TEXT'},
            '2017T3': {'closed': 'Q2.3', 'open': 'Q2.3_5_TEXT'},
            '2018T1': {'closed': 'Q2.3', 'open': 'Q2.3_5_TEXT'},
            '2018T3': {'closed': 'Q2.3', 'open': 'Q2.3_5_TEXT'},
            '2019T1': {'closed': 'Q2.3', 'open': 'Q2.3.1'},
            '2021T1': {'closed': 'Q2.3', 'open': 'Q2.3_5_TEXT'},
            '2022T1': {'closed': 'Q2.3', 'open': 'Q2.3.1'}
        }

        self.course_survey_mapping = {
            "EX101x_1T2015": "No results collected",
            "EX101x_1T2016": "No results collected",
            "EX101x_2T2018": "2018T1",
            "EX101x_3T2017": "2017T3",
            "EX101x_3T2015": "2015T3",
            "EX101x_3T2016": "2016T3",
            "FP101x_3T2014": "No results collected",
            "FP101x_3T2015": "2015T3",
            "ST1x_3T2018": "2018T3",
            "ST1x_3T2019": "2019T1",
            "ST1x_3T2020": "2019T1",
            "ST1x_3T2021": "2021T1",
            "ST1x_3T2022": "2022T1",
            "UnixTx_1T2020": "2019T1",
            "UnixTx_1T2022": "2022T1",
            "UnixTx_2T2021": "2021T1",
            "UnixTx_3T2020": "2019T1"
        }

    def get_closed_question(self, survey_version: str):
        """Retrieve the closed question ID based on survey version."""
        return self.survey_data.get(survey_version, {}).get('closed', 'Question ID not found')

    def get_open_question(self, survey_version: str):
        """Retrieve the open question ID based on survey version."""
        return self.survey_data.get(survey_version, {}).get('open', 'Question ID not found')

    def get_survey_version(self, course_id: str):
        """Retrieve the survey version based on course ID."""
        return self.course_survey_mapping.get(course_id, 'Survey version not found')


def find_all_presurveys_for_course_id(course_id: str) -> list[str]:
    return glob.glob(f"{WORKING_DIRECTORY}/{course_id}*/pre_survey*.txt", recursive=True)


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    tokens = nltk.word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    processed_text = ' '.join(stemmed_tokens)
    return processed_text


def cluster_responses():
    load_dotenv()
    survey_questions = SurveyQuestions()
    course_contingency_tables = {}

    for course in COURSES:
        presurveys = find_all_presurveys_for_course_id(course)
        if not presurveys:
            print(f"No presurveys found for course {course}")
            continue

        tables = []

        for presurvey_file in presurveys:
            parts = presurvey_file.rstrip('.txt').split('_')
            course_run = '_'.join([parts[-2], parts[-1]])
            survey_version = survey_questions.get_survey_version(course_run)
            closed_question = survey_questions.get_closed_question(
                survey_version)
            open_question = survey_questions.get_open_question(survey_version)
            presurvey = pd.read_csv(presurvey_file, sep='\t')
            presurvey = presurvey.drop(0)
            user_ids = presurvey["hash_id"].tolist()
            genders = find_genders_by_user_ids(user_ids)
            presurvey["gender"] = presurvey["hash_id"].map(genders)

            presurvey = presurvey.rename(columns={closed_question: 'response'})
            closed_question = 'response'

            contingency_table = create_contingency_table_for_run(
                presurvey, closed_question)

            write_clusters_to_csv(
                presurvey, closed_question, course_run)

            tables.append(contingency_table)

            print(f"Processing {course} survey version {survey_version}")
            open_question_gender_and_hash_id = presurvey[[
                open_question, 'gender', 'hash_id']].dropna()
            save_open_question_responses_to_file(
                open_question_gender_and_hash_id, course)
        if tables:
            combined_table = pd.concat(tables).groupby(level=0).sum()
            course_contingency_tables[course] = combined_table

    # Perform Chi-squared tests for each course
    for course, table in course_contingency_tables.items():
        # Find all csv files in reasons_for_enrolment/clusters with the same course name in them and combine them into one.
        cluster_files = glob.glob(
            f"reasons_for_enrolment/clusters/{course}*.csv")
        cluster_dfs = [pd.read_csv(file) for file in cluster_files]
        combined_clusters = pd.concat(cluster_dfs)
        combined_clusters = combined_clusters.dropna(axis=1, how='all')

        # Exclude the head and remove any trailing commas
        combined_clusters.to_csv(
            f"reasons_for_enrolment/clusters/{course}.csv", index=False, header=True)

        # Remove all the files that were combined
        for file in cluster_files:
            os.remove(file)

        table = table.loc[(
            table['Male'] != 0) | (table['Female'] != 0)]
        merged_table = merge_contingency_table_answers(table)
        merged_table.to_csv(
            f"reasons_for_enrolment/closed_responses/{course}_contingency_table.csv")
        chi2, p, dof, expected = chi2_contingency(merged_table)
        contingency_table_percentages = merged_table.div(
            merged_table.sum(axis=0), axis=1) * 100
        contingency_table_percentages = contingency_table_percentages.round(2)
        print(contingency_table_percentages)
        print(f"Results of Chi-squared test for {course}:")
        print(f"Chi-squared Statistic: {chi2}, P-value: {p}")


def create_contingency_table_for_run(presurvey: pd.DataFrame, closed_question: str) -> pd.DataFrame:
    male_df = presurvey[presurvey['gender'] == 'm']
    female_df = presurvey[presurvey['gender'] == 'f']

    male_counts = male_df[closed_question].value_counts().reindex(
        presurvey[closed_question].unique(), fill_value=0)
    female_counts = female_df[closed_question].value_counts().reindex(
        presurvey[closed_question].unique(), fill_value=0)

    contingency_table = pd.DataFrame({
        'Male': male_counts,
        'Female': female_counts
    }).dropna()

    return contingency_table


def save_open_question_responses_to_file(open_question_responses: pd.DataFrame, course_run: str) -> None:
    open_question_responses.to_csv(
        f"reasons_for_enrolment/open_responses/{course_run}_open_question_responses.csv", index=False)


def write_clusters_to_csv(presurvey_df: pd.DataFrame, closed_question: str, course_run: str) -> None:
    presurvey_df = presurvey_df[['hash_id', 'gender', closed_question]]
    presurvey_df = presurvey_df.dropna(subset=[closed_question])
    presurvey_df[closed_question] = presurvey_df[closed_question].apply(
        lambda x: 'know the instructor' if 'instructor' in x.lower() else x)
    presurvey_df[closed_question] = presurvey_df[closed_question].apply(
        lambda x: 'career' if 'career' in x.lower() or 'job' in x.lower() or 'work' in x.lower() else x)
    presurvey_df[closed_question] = presurvey_df[closed_question].apply(
        lambda x: 'degree' if 'degree' in x.lower() or 'studies' in x.lower() else x)
    presurvey_df[closed_question] = presurvey_df[closed_question].apply(
        lambda x: 'other' if 'other' in x.lower() else x)
    presurvey_df[closed_question] = presurvey_df[closed_question].apply(
        lambda x: 'teaching' if 'teach' in x.lower() else x)
    presurvey_df[closed_question] = presurvey_df[closed_question].apply(
        lambda x: 'interest' if 'interest' in x.lower() else x)

    presurvey_df = presurvey_df[presurvey_df['gender'].isin(['m', 'f'])]
    presurvey_df.to_csv(
        f"reasons_for_enrolment/clusters/{course_run}.csv", index=False)


def merge_contingency_table_answers(contingency_table: pd.DataFrame) -> pd.DataFrame:
    """Merge contingency table answers."""

    print(contingency_table)
    instructor_indices = [
        idx for idx in contingency_table.index if 'instructor' in idx.lower()]
    if instructor_indices:
        instructor_data = contingency_table.loc[instructor_indices].sum()
        contingency_table = contingency_table.drop(instructor_indices)
        contingency_table.loc['know the instructor'] = instructor_data

    career_indices = [
        idx for idx in contingency_table.index if 'career' in idx.lower() or 'job' in idx.lower() or 'work' in idx.lower()
    ]
    if career_indices:
        career_data = contingency_table.loc[career_indices].sum()
        contingency_table = contingency_table.drop(career_indices)
        contingency_table.loc['career'] = career_data

    studies_indices = [
        idx for idx in contingency_table.index if 'degree' in idx.lower() or 'studies' in idx.lower()
    ]
    if studies_indices:
        degree_data = contingency_table.loc[studies_indices].sum()
        contingency_table = contingency_table.drop(studies_indices)
        contingency_table.loc['degree'] = degree_data

    other_indices = [
        idx for idx in contingency_table.index if 'other' in idx.lower()
    ]
    if other_indices:
        other_data = contingency_table.loc[other_indices].sum()
        contingency_table = contingency_table.drop(other_indices)
        contingency_table.loc['other'] = other_data

    teaching_indices = [
        idx for idx in contingency_table.index if 'teach' in idx.lower()
    ]
    if teaching_indices:
        teaching_data = contingency_table.loc[teaching_indices].sum()
        contingency_table = contingency_table.drop(teaching_indices)
        contingency_table.loc['teaching'] = teaching_data

    interested_indices = [
        idx for idx in contingency_table.index if 'interest' in idx.lower()
    ]
    if interested_indices:
        interested_data = contingency_table.loc[interested_indices].sum()
        contingency_table = contingency_table.drop(interested_indices)
        contingency_table.loc['interest'] = interested_data

    return contingency_table
