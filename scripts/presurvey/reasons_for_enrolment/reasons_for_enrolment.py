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
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numerals
    text = text.translate(str.maketrans('', '', string.digits))

    # Tokenize text
    tokens = nltk.word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Join tokens back to string
    processed_text = ' '.join(stemmed_tokens)
    return processed_text


def cluster_responses():
    load_dotenv()
    COURSES = json.loads(os.getenv("COURSES"))
    survey_questions = SurveyQuestions()
    for course in COURSES:
        presurveys = find_all_presurveys_for_course_id(course)
        if not presurveys:
            print(f"No presurveys found for course {course}")
            continue
        for presurvey_file in presurveys:
            parts = presurvey_file.rstrip('.txt').split('_')
            course_run = '_'.join([parts[-2], parts[-1]])

            print(course_run)

            pre_survey = pd.read_csv(presurvey_file, sep='\t')
            pre_survey = pre_survey.drop(0)

            user_ids = pre_survey["hash_id"].tolist()
            genders = find_genders_by_user_ids(user_ids)

            pre_survey["gender"] = pre_survey["hash_id"].map(genders)

            male_df = pre_survey[pre_survey['gender'] == 'm']
            female_df = pre_survey[pre_survey['gender'] == 'f']

            survey_version = survey_questions.get_survey_version(course_run)
            closed_question = survey_questions.get_closed_question(
                survey_version)

            closed_question_percentages_male = male_df[closed_question].value_counts(
                normalize=True) * 100
            # print(f"""Closed question responses for male: {
            #       len(male_df[closed_question])}""")
            # print(f"""Closed question response percentages male {
            #       closed_question_percentages_male}""")
            # close_question_percentages_female = female_df[closed_question].value_counts(
            #     normalize=True) * 100
            # print(f"""Closed question responses for female: {
            #       len(female_df[closed_question])}""")
            # print(f"""Closed question response percentages female {
            #       close_question_percentages_female}""")

            male_counts = male_df[closed_question].value_counts().reindex(
                pre_survey[closed_question].unique(), fill_value=0)
            female_counts = female_df[closed_question].value_counts().reindex(
                pre_survey[closed_question].unique(), fill_value=0)
            # Create a contingency table
            contingency_table = pd.DataFrame({
                'Male': male_counts,
                'Female': female_counts
            })

            contingency_table = contingency_table.loc[(
                contingency_table['Male'] != 0) | (contingency_table['Female'] != 0)]

            print("Contingency Table:")
            # print(contingency_table)

            # Perform the chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            print(f"Chi-squared Test statistic: {chi2}")
            print(f"P-value: {p_value}")
            print(f"Degrees of freedom: {dof}")
            print("Expected frequencies:")
            # print(pd.DataFrame(expected, columns=contingency_table.columns,
            #   index=contingency_table.index))

            # Evaluate the p-value
            if p_value < 0.05:
                print(
                    "There is a statistically significant difference between male and female responses.")
            else:
                print(
                    "There is no statistically significant difference between male and female responses.")

            # open_question = survey_questions.get_open_question(survey_version)
            # # Drop NaN from open_question
            # pre_survey = pre_survey.dropna(subset=[open_question])
            # pre_survey[open_question] = pre_survey[open_question].apply(
            #     preprocess_text)

            # # Print pre_survey[open_question] length
            # print(f"Length of open question responses: {
            #       len(pre_survey[open_question])}")

            # if len(pre_survey[open_question]) < 10:
            #     print(f"Skipping course {
            #           course_run} due to insufficient responses.")
            #     continue

            # vectorizer = TfidfVectorizer(
            #     max_df=0.5, min_df=0.01, ngram_range=(1, 2))
            # tfidf_matrix = vectorizer.fit_transform(pre_survey[open_question])

            # feature_names = vectorizer.get_feature_names_out()
            # unique_terms_count = len(feature_names)
            # print(f"Unique terms count: {unique_terms_count}")

            # num_clusters = 5
            # km = KMeans(n_clusters=num_clusters, init='k-means++',
            #             max_iter=300, n_init=10)
            # km.fit(tfidf_matrix)
            # clusters = km.labels_.tolist()

            # Evaluation
            # print("\nTop terms per cluster:")
            # order_centroids = km.cluster_centers_.argsort()[:, ::-1]
            # features = vectorizer.get_feature_names_out()
            # for i in range(num_clusters):
            #     print(f"Cluster {i}: ", end='')
            #     for ind in order_centroids[i, :10]:  # Top 10 terms
            #         print(f'{features[ind]} ', end='')
            #     print()

            # # Silhouette Score
            # score = silhouette_score(tfidf_matrix, clusters)
            # print("Silhouette Score: ", score)
