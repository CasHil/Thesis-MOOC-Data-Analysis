import re
from typing import Tuple
import traceback
from collections import Counter
from functools import partial

import pandas as pd
import dask.dataframe as dd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pymongo import MongoClient
from pymongo.database import Database
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_courses(db: Database) -> pd.DataFrame:
    courses = db["courses"].find()
    courses_df = pd.DataFrame(courses)

    if courses_df.empty:
        raise ValueError("No courses found.")

    courses_df['start_time'] = pd.to_datetime(courses_df['start_time'])
    courses_df['end_time'] = pd.to_datetime(courses_df['end_time'])

    return courses_df


def get_metadata(db: Database, course_id: str) -> pd.DataFrame:
    metadata_query = {
        "name": {
            "$regex": course_id,
            "$options": "i"

        }
    }
    metadata = db["metadata"].find_one(metadata_query)
    metadata_df = pd.DataFrame(metadata)

    if metadata_df.empty:
        raise ValueError(f"No metadata found for course {course_id}")

    return metadata_df


def has_due_dates(metadata: pd.DataFrame) -> bool:
    has_due_dates = False
    element_time_map_due_series = pd.Series(
        metadata["object"]["element_time_map_due"])
    for _, value in element_time_map_due_series.items():
        unix_epoch = pd.Timestamp(
            "1970-01-01T00:00:00.000+00:00").tz_localize(None)
        due_date = pd.Timestamp(value).tz_localize(None)
        if due_date and due_date != unix_epoch:
            has_due_dates = True
            break

    if not has_due_dates:
        raise ValueError(
            f"Course {metadata["object"]["course_name"]} does not have assignments with due dates.")
    return has_due_dates


def get_learner_demographic(db: Database, course_id: str) -> pd.DataFrame:
    learner_demographic_query = {
        "course_learner_id": {
            "$regex": course_id,
            "$options": "i"
        }
    }
    learner_demographic = db["learner_demographic"].find(
        learner_demographic_query)

    learner_demographic_df = pd.DataFrame(learner_demographic)

    if learner_demographic_df.empty:
        raise ValueError(
            f"No learner demographic data found for course {course_id}")

    return downcast_numeric_columns(learner_demographic)


def get_learner_demographic(db: Database, course_id: str) -> pd.DataFrame:
    learner_demographic_query = {
        "course_learner_id": {
            "$regex": course_id,
            "$options": "i"
        }
    }
    learner_demographic = db["learner_demographic"].find(
        learner_demographic_query)

    learner_demographic_df = pd.DataFrame(learner_demographic)

    if learner_demographic_df.empty:
        raise ValueError(
            f"No learner demographic data found for course {course_id}")

    return pd.DataFrame(learner_demographic_df)


def process_element_id(x: str) -> str:
    return "+type@".join(x.split("+type@")[1:])


def escape_id(course_id: str) -> str:
    return re.escape(course_id)


def get_submissions(db: Database, course_id: str) -> pd.DataFrame:
    submissions_query = {
        "submission_id": {
            "$regex": course_id,
            "$options": "i"
        },
        "question_id": {
            "$ne": None
        }
    }
    submissions = db["submissions"].find(submissions_query)
    submissions_df = pd.DataFrame(submissions)
    if submissions_df.empty:
        print(
            f"No submissions found for course {course_id}")
        return pd.DataFrame()

    submissions_df = submissions_df[~submissions_df['submission_id'].str.contains(
        "undefined")]

    submissions_df['submission_timestamp'] = pd.to_datetime(
        submissions_df['submission_timestamp'])
    return submissions_df


def get_chapter_start_time_by_video_id(metadata_df: pd.DataFrame, video_id: str) -> pd.Timestamp:
    child_parent_map = metadata_df["object"]["child_parent_map"]
    matching_keys = [key for key in child_parent_map if video_id in key]
    block_id = metadata_df["object"]["child_parent_map"][matching_keys[0]]
    section_id = child_parent_map[block_id]
    chapter_id = child_parent_map[section_id]
    return pd.Timestamp(metadata_df["object"]["element_time_map"][chapter_id])


def get_video_interactions(db: Database, course_id: str) -> pd.DataFrame:
    video_interactions_query = {
        "course_learner_id": {
            "$regex": course_id,
            "$options": "i"
        }
    }
    video_interactions = db["video_interactions"].find(
        video_interactions_query)

    video_interactions_df = pd.DataFrame(video_interactions)
    if video_interactions_df.empty:
        print(
            f"No video interactions found for course {course_id}")
        return pd.DataFrame()

    video_interactions_df['start_time'] = pd.to_datetime(
        video_interactions_df['start_time'])
    video_interactions_df['end_time'] = pd.to_datetime(
        video_interactions_df['end_time'])

    return downcast_numeric_columns(video_interactions_df)


def downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    # https://stackoverflow.com/questions/65842209/how-to-downcast-numeric-columns-in-pandas
    # Downcast for memory efficiency
    df_copy = df.copy()
    fcols = df_copy.select_dtypes('float').columns
    icols = df_copy.select_dtypes('integer').columns
    df_copy[fcols] = df_copy[fcols].apply(
        pd.to_numeric, downcast='float')
    df_copy[icols] = df_copy[icols].apply(
        pd.to_numeric, downcast='integer')
    return df_copy


def get_course_learner(db: Database, course_id: str) -> pd.DataFrame:
    course_learner_query = {
        "course_learner_id": {
            "$regex": course_id,
            "$options": "i"
        }
    }

    course_learners = db["course_learner"].find(course_learner_query)
    course_learners_df = pd.DataFrame(course_learners)

    if course_learners_df.empty:
        raise ValueError(f"No course learners found for course {course_id}")

    return course_learners_df


def get_ora_sessions(db: Database, course_id: str) -> pd.DataFrame:
    ora_sessions_query = {
        "course_learner_id": {
            "$regex": course_id,
            "$options": "i"
        }
    }
    ora_sessions = db["ora_sessions"].find(ora_sessions_query)
    ora_sessions_df = pd.DataFrame(ora_sessions)

    if ora_sessions_df.empty:
        print(
            f"No ORA sessions found for course {course_id}")
        return pd.DataFrame()

    ora_sessions_df['start_time'] = pd.to_datetime(
        ora_sessions_df['start_time'])
    ora_sessions_df['end_time'] = pd.to_datetime(
        ora_sessions_df['end_time'])

    return downcast_numeric_columns(ora_sessions_df)


def process_element_id(x: str) -> str:
    return "+type@".join(x.split("+type@")[1:])


def get_learner_submissions_per_assessment_period(learner_submissions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    within_period = learner_submissions['submission_timestamp'].between(
        assessment_period.left, assessment_period.right)
    return learner_submissions[within_period]


def get_videos_per_assessment_period(learner_video_interactions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    within_period = learner_video_interactions['end_time'].between(
        assessment_period.left, assessment_period.right)
    return learner_video_interactions[within_period]


def engagement_by_submissions(submissions_by_learner: pd.DataFrame, due_dates_per_assessment_period: pd.Series, learner_id: str) -> pd.Series:
    submissions_for_learner = submissions_by_learner[
        submissions_by_learner['course_learner_id'] == learner_id]

    elements_completed_on_time = set()
    elements_completed_late = set()
    engagement = pd.Series()

    for idx, assessment_period in enumerate(due_dates_per_assessment_period.index):
        elements_due_in_period = set(
            due_dates_per_assessment_period[assessment_period])

        # If there are no submissions due, it should not count for the learner's engagement
        if not any(["problem+block@" in element for element in elements_due_in_period]):
            continue
        period_submissions = get_learner_submissions_per_assessment_period(
            submissions_for_learner, assessment_period)
        for element in elements_due_in_period:
            element_in_period_submissions = any(
                period_submissions['question_id'].str.contains(element))
            if element_in_period_submissions:
                elements_completed_on_time.add(element)
            else:
                elements_completed_late.add(element)

        if elements_due_in_period.issubset(elements_completed_on_time):
            engagement[idx] = "T"
        elif elements_due_in_period.issubset(elements_completed_on_time.union(elements_completed_late)):
            engagement[idx] = "B"
        elif not elements_due_in_period.isdisjoint(elements_completed_on_time.union(elements_completed_late)):
            engagement[idx] = "A"
        else:
            engagement[idx] = "O"
    return engagement


def engagement_by_video_interactions(video_interactions: pd.DataFrame, due_dates_per_assessment_period: pd.Series, metadata_df: pd.DataFrame, learner_id: str) -> pd.Series:
    video_interactions_copy = \
        video_interactions[video_interactions['course_learner_id']
                           == learner_id].copy()

    has_watched_video_in_assessment_period = set()
    engagement = pd.Series()

    video_interactions_copy['chapter_start_time'] = video_interactions_copy['video_id'].apply(
        lambda video_id: get_chapter_start_time_by_video_id(metadata_df, video_id))
    chapter_to_assessment_period = {
        period.left: period for period in due_dates_per_assessment_period.index}

    video_interactions_copy['assessment_period'] = video_interactions_copy['chapter_start_time'].map(
        chapter_to_assessment_period)
    valid_video_interactions = video_interactions_copy.dropna(
        subset=['assessment_period'])
    has_watched_video_in_assessment_period.update(
        valid_video_interactions['assessment_period'].unique())

    for assessment_period in has_watched_video_in_assessment_period:
        engagement[assessment_period] = "A"

    return engagement


def engagement_by_quiz_sessions(quiz_sessions: pd.DataFrame, quizzes_due_per_assessment_period: pd.Series, learner_id: str) -> pd.Series:
    quiz_sessions_for_learner = quiz_sessions[quiz_sessions['course_learner_id'] == learner_id]
    assessment_periods_with_all_quizzes_on_time = set()
    assessment_periods_with_all_quizzes_late_or_on_time = set()
    assessment_periods_with_some_quizzes_done = set()

    engagement = pd.Series()

    for assessment_period, elements_due in quizzes_due_per_assessment_period.items():
        quiz_sessions_in_period = get_learner_quiz_sessions_per_assessment_period(
            quiz_sessions_for_learner, assessment_period)
        all_quizzes_done_on_time = True
        all_quizzes_done = True
        some_quizzes_done = False
        for element in elements_due:
            element_in_period_quiz_sessions = any(
                quiz_sessions_in_period['sessionId'].str.contains(element))
            if not element_in_period_quiz_sessions:
                all_quizzes_done_on_time = False
                continue
            else:
                some_quizzes_done = True

            element_in_all_quiz_sessions = any(
                quiz_sessions_for_learner['sessionId'].str.contains(element))
            if not element_in_all_quiz_sessions:
                all_quizzes_done = False
            else:
                some_quizzes_done = True
        if all_quizzes_done_on_time:
            assessment_periods_with_all_quizzes_on_time.add(assessment_period)
        elif all_quizzes_done:
            assessment_periods_with_all_quizzes_late_or_on_time.add(
                assessment_period)
        elif some_quizzes_done:
            assessment_periods_with_some_quizzes_done.add(assessment_period)

    for assessment_period in quizzes_due_per_assessment_period.index:
        if assessment_period in assessment_periods_with_all_quizzes_on_time:
            engagement[assessment_period] = "T"
        elif assessment_period in assessment_periods_with_all_quizzes_late_or_on_time:
            engagement[assessment_period] = "B"
        elif assessment_period in assessment_periods_with_some_quizzes_done:
            engagement[assessment_period] = "A"
        elif quizzes_due_per_assessment_period.get(assessment_period, None) is not None:
            engagement[assessment_period] = "O"
        else:
            engagement[assessment_period] = None

    return engagement


def engagement_by_ora_sessions(ora_sessions: pd.DataFrame, due_dates_per_assessment_period: pd.Series, learner_id: str) -> pd.Series:
    ora_sessions_for_learner = ora_sessions[ora_sessions['course_learner_id'] == learner_id]
    ora_due_dates_per_assessment_period = due_dates_per_assessment_period.apply(
        lambda elements: [element for element in elements if "openassessment" in element])
    assessment_periods_with_all_oras_on_time = set()
    assessment_periods_with_all_oras_late_or_on_time = set()
    assessment_periods_with_some_oras_done = set()

    engagement = pd.Series()

    for assessment_period, elements_due in ora_due_dates_per_assessment_period.items():
        ora_sessions_in_period = get_learner_ora_sessions_per_assessment_period(
            ora_sessions_for_learner, assessment_period)
        all_oras_done_on_time = True
        all_oras_done = True
        some_oras_done = False
        for element in elements_due:
            element_in_period_ora_sessions = any(
                ora_sessions_in_period['sessionId'].str.contains(element))
            if not element_in_period_ora_sessions:
                all_oras_done_on_time = False
                continue
            else:
                some_oras_done = True
            element_in_all_ora_sessions = any(
                ora_sessions_for_learner['assessment_id'].str.contains(element))
            if not element_in_all_ora_sessions:
                all_oras_done = False
            else:
                some_oras_done = True
        if all_oras_done_on_time:
            assessment_periods_with_all_oras_on_time.add(assessment_period)
        elif all_oras_done:
            assessment_periods_with_all_oras_late_or_on_time.add(
                assessment_period)
        elif some_oras_done:
            assessment_periods_with_some_oras_done.add(assessment_period)

    for idx, assessment_period in enumerate(ora_due_dates_per_assessment_period.index):
        if assessment_period in assessment_periods_with_all_oras_on_time:
            engagement[idx] = "T"
        elif assessment_period in assessment_periods_with_all_oras_late_or_on_time:
            engagement[idx] = "B"
        elif assessment_period in assessment_periods_with_some_oras_done:
            engagement[idx] = "A"
        elif ora_due_dates_per_assessment_period.get(assessment_period, None) is not None:
            engagement[idx] = "O"
        else:
            engagement[idx] = None

    return engagement


def construct_learner_engagement_mapping(quiz_sessions_df: pd.DataFrame, metadata_df: pd.DataFrame, course_learner_df: pd.DataFrame, video_interactions_df: pd.DataFrame, submissions_df: pd.DataFrame, ora_sessions_df: pd.DataFrame, learner_demographic_df: pd.DataFrame) -> Tuple[Counter[tuple, int], Counter[tuple, int]]:
    course_start_date = pd.Timestamp(metadata_df["object"]["start_date"])
    due_dates = {key: pd.Timestamp(
        value) for key, value in metadata_df["object"]["element_time_map_due"].items()}
    child_parent_map = metadata_df["object"]["child_parent_map"]

    valid_course_learner_ids = set(course_learner_df['course_learner_id'])
    video_interactions_df = video_interactions_df[video_interactions_df['course_learner_id'].isin(
        valid_course_learner_ids)]
    quiz_sessions_df = quiz_sessions_df[quiz_sessions_df['course_learner_id'].isin(
        valid_course_learner_ids)]
    ora_sessions_df = ora_sessions_df[ora_sessions_df['course_learner_id'].isin(
        valid_course_learner_ids)]
    submissions_df = submissions_df[submissions_df['course_learner_id'].isin(
        valid_course_learner_ids)]

    flattened_due_dates: dict = due_dates.copy()
    for element in due_dates:
        if "sequential" in element:
            element_children = {
                key: due_dates[element] for key in child_parent_map if child_parent_map[key] == element}
            del flattened_due_dates[element]
            flattened_due_dates = flattened_due_dates | element_children

    due_dates_per_assessment_period: pd.Series = get_due_dates_per_assessment_period(
        flattened_due_dates, course_start_date)

    quizzes_due_per_assessment_period = due_dates_per_assessment_period.apply(
        lambda elements: [element for element in elements if "vertical+block" in element] or None).dropna()
    submissions_due_per_assessment_period = due_dates_per_assessment_period.apply(
        lambda elements: [element for element in elements if "problem+block" in element] or None).dropna()
    ora_due_per_assessment_period = due_dates_per_assessment_period.apply(
        lambda elements: [element for element in elements if "openassessment" in element] or None).dropna()

    tqdm.pandas()

    course_learner_df['quiz_engagement'] = course_learner_df['course_learner_id'].progress_apply(
        lambda learner_id: engagement_by_quiz_sessions(quiz_sessions_df, quizzes_due_per_assessment_period, learner_id))
    course_learner_df['video_engagement'] = course_learner_df['course_learner_id'].progress_apply(
        lambda learner_id: engagement_by_video_interactions(video_interactions_df, due_dates_per_assessment_period, metadata_df, learner_id))
    course_learner_df['submission_engagement'] = course_learner_df['course_learner_id'].progress_apply(
        lambda learner_id: engagement_by_submissions(submissions_df, submissions_due_per_assessment_period, learner_id))
    course_learner_df['ora_engagement'] = course_learner_df['course_learner_id'].progress_apply(
        lambda learner_id: engagement_by_ora_sessions(ora_sessions_df, ora_due_per_assessment_period, learner_id))

    course_learner_df['engagement'] = course_learner_df.progress_apply(
        lambda row: process_engagement(row['quiz_engagement'], row['video_engagement'], row['submission_engagement'], row['ora_engagement'], due_dates_per_assessment_period
                                       ), axis=1)

    male_counts = Counter(course_learner_df[course_learner_df['gender']
                                            == 'm']['engagement'].value_counts())
    female_counts = Counter(course_learner_df[course_learner_df['gender']
                                              == 'f']['engagement'].value_counts())

    return male_counts, female_counts


def process_engagement(quiz_engagement: pd.Series, video_engagement: pd.Series, submission_engagement: pd.Series, ora_engagement: pd.Series, assessment_periods: pd.Series) -> pd.Series:
    engagement = pd.Series()
    for assessment_period in assessment_periods:
        quiz_engagement_for_period = quiz_engagement.get(assessment_period, "")
        video_engagement_for_period = video_engagement.get(
            assessment_period, "")
        submission_engagement_for_period = submission_engagement.get(
            assessment_period, "")
        ora_engagement_for_period = ora_engagement.get(assessment_period, "")

        non_video_statuses = set([quiz_engagement_for_period,
                                  submission_engagement_for_period, ora_engagement_for_period])

        non_video_statuses.discard("")

        # Rule 1: If any status is 'A', return 'A'
        if "A" in non_video_statuses:
            return "A"
        # Rule 2: If it's a mix of 'T' and 'B', or just 'B', return 'B'
        elif "B" in non_video_statuses and non_video_statuses == {"T", "B"}:
            return "B"
        # Rule 3: If all are 'T', return 'T'
        elif non_video_statuses == {"T"}:
            return "T"
        # Rule 4: If all non-video engagements are "", check video engagement
        elif not non_video_statuses:
            # If video_engagement is 'A', return 'A'. Otherwise, return 'O' since video_engagement can only be 'A' or 'O'.
            return video_engagement_for_period if video_engagement_for_period == "A" else "O"
        # Default case to catch any unforeseen combinations, primarily for safety
        return "O"

    return engagement


def get_learner_quiz_sessions_per_assessment_period(learner_quiz_sessions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    within_period = learner_quiz_sessions['end_time'].between(
        assessment_period.left, assessment_period.right)
    return learner_quiz_sessions[within_period]


def get_learner_ora_sessions_per_assessment_period(learner_ora_sessions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    within_period = learner_ora_sessions['end_time'].between(
        assessment_period.left, assessment_period.right)
    return learner_ora_sessions[within_period]


def get_due_dates_per_assessment_period(element_time_map_due: dict, course_start_date: pd.Timestamp) -> pd.Series:
    temp_df = pd.DataFrame(list(element_time_map_due.items()), columns=[
        'block_id', 'due_date'])

    temp_df['due_date'] = pd.to_datetime(temp_df['due_date'])

    sorted_df = temp_df.sort_values(by='due_date')

    grouped = sorted_df.groupby('due_date')['block_id'].apply(list)

    grouped = grouped.sort_index()
    periods_with_due_dates = pd.Series(index=pd.IntervalIndex.from_arrays(
        [course_start_date] + grouped.index[:-1].tolist(), grouped.index.tolist()), data=grouped.values)

    return periods_with_due_dates


def engagement_to_numeric(engagement: tuple) -> list:
    label_to_numeric = {'T': 3, 'B': 2, 'A': 1, 'O': 0}
    return [label_to_numeric[label] for label in engagement]


def calculate_k_means_clusters(learner_engagement_mapping: Counter[tuple, int], title: str) -> None:
    engagement_numeric = [engagement_to_numeric(
        pattern) for pattern in learner_engagement_mapping.keys()]
    print(engagement_numeric)
    X = np.array(engagement_numeric)

    best_score = -1
    best_kmeans = None

    for _ in range(100):
        kmeans = KMeans(n_clusters=4, random_state=None).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_kmeans = kmeans

    clusters = best_kmeans.labels_
    centroids = best_kmeans.cluster_centers_

    df = pd.DataFrame(X, columns=[f'Week {i+1}' for i in range(X.shape[1])])
    df['cluster'] = clusters

    print(f"\n{title}")
    print(f"Best silhouette score: {best_score:.4f}")

    print("\nCluster sizes:")
    print(df['cluster'].value_counts().sort_index())

    for cluster in sorted(df['cluster'].unique()):
        print(f"\nAverage engagement scores for Cluster {cluster}:")
        print(df[df['cluster'] == cluster].iloc[:, :-1].mean())

    plt.figure(figsize=(10, 6))
    for i, centroid in enumerate(centroids):
        plt.plot(centroid, label=f'Cluster {i}', marker='o', linestyle='--')
    plt.title(f'Cluster Centroids for {title}')
    plt.xlabel('Week')
    plt.ylabel('Engagement Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(X.shape[1]), [f'Week {i+1}' for i in range(X.shape[1])])
    plt.savefig(f'{title}.png')


def get_quiz_sessions(db: Database, course_id: str) -> pd.DataFrame:
    quiz_sessions_query = {
        "course_learner_id": {
            "$regex": course_id,
            "$options": "i"
        }
    }
    quiz_sessions = db["quiz_sessions"].find(quiz_sessions_query)
    quiz_sessions_df = pd.DataFrame(quiz_sessions)

    if quiz_sessions_df.empty:
        print(
            f"No quiz sessions found for course {course_id}")
        return pd.DataFrame()

    quiz_sessions_df['start_time'] = pd.to_datetime(
        quiz_sessions_df['start_time'])
    quiz_sessions_df['end_time'] = pd.to_datetime(
        quiz_sessions_df['end_time'])

    return downcast_numeric_columns(quiz_sessions_df)


def main() -> None:
    client = MongoClient("mongodb://localhost:27017/")
    print("Connected to MongoDB")
    db = client["edx_testing"]

    courses = get_courses(db)

    for _, course in courses.iterrows():
        try:
            full_course_id = course["course_id"]
            escaped_course_id = escape_id(full_course_id)
            mooc_id = escaped_course_id.split("+")[-2]
            course_run = escaped_course_id.split("+")[-1]
            course_id = f"{mooc_id}_{course_run}"
            metadata = get_metadata(db, course_id)

            if not has_due_dates(metadata):
                raise ValueError("Course does not have due dates.")

            submissions = get_submissions(db, escaped_course_id)
            print("Submissions done")
            video_interactions = get_video_interactions(db, escaped_course_id)
            print("Video interactions done")
            quiz_sessions = get_quiz_sessions(db, escaped_course_id)
            print("Quiz sessions done")
            course_learner = get_course_learner(db, escaped_course_id)
            ora_sessions = get_ora_sessions(db, escaped_course_id)

            course_learner = course_learner.merge(get_learner_demographic(
                db, escaped_course_id)[['course_learner_id', 'gender']], on='course_learner_id', how='left')
            filtered_learners = course_learner[course_learner['gender'].isin([
                'm', 'f'])]

            filtered_ids = filtered_learners['course_learner_id'].unique()

            if not submissions.empty:
                submissions = submissions[submissions['course_learner_id'].isin(
                    filtered_ids)]

            if not video_interactions.empty:
                video_interactions = video_interactions[video_interactions['course_learner_id'].isin(
                    filtered_ids)]

            if not quiz_sessions.empty:
                quiz_sessions = quiz_sessions[quiz_sessions['course_learner_id'].isin(
                    filtered_ids)]

            if not ora_sessions.empty:
                ora_sessions = ora_sessions[ora_sessions['course_learner_id'].isin(
                    filtered_ids)]

            if not course_learner.empty:
                course_learner = course_learner[course_learner['course_learner_id'].isin(
                    filtered_ids)]

            print("All data loaded")
            male_learner_engagement_mapping, female_learner_engagement_mapping = construct_learner_engagement_mapping(quiz_sessions, metadata,
                                                                                                                      course_learner, video_interactions, submissions, ora_sessions, filtered_learners)
            calculate_k_means_clusters(
                male_learner_engagement_mapping, course_id)
            calculate_k_means_clusters(
                female_learner_engagement_mapping, course_id)

        except Exception as ex:
            print(''.join(traceback.format_exception(type(ex),
                                                     value=ex, tb=ex.__traceback__)))


if __name__ == '__main__':
    main()
