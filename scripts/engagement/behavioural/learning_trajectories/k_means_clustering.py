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


def construct_query(field: str, regex: str, filtered_ids: list[str] = None):
    query = {
        field: {
            "$regex": regex,
            "$options": "i"
        }
    }

    if filtered_ids is not None:
        query["course_learner_id"] = {
            "$in": filtered_ids
        }

    return query


def get_metadata(db: Database, course_id: str) -> pd.DataFrame:
    metadata_query = construct_query("name", course_id)
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

    return has_due_dates


def get_learner_demographic(db: Database, course_id: str, filtered_ids: list[str]) -> pd.DataFrame:
    learner_demographic_query = construct_query(
        "course_learner_id", course_id, filtered_ids)
    learner_demographic = db["learner_demographic"].find(
        learner_demographic_query)

    learner_demographic_df = pd.DataFrame(learner_demographic)

    if learner_demographic_df.empty:
        raise ValueError(
            f"No learner demographic data found for course {course_id}")

    return downcast_numeric_columns(learner_demographic)


def get_learner_demographic(db: Database, course_id: str) -> pd.DataFrame:
    learner_demographic_query = construct_query(
        "course_learner_id", course_id)
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


def get_submissions(db: Database, course_id: str, filtered_ids: list[str]) -> pd.DataFrame:
    submissions_query = {
        "submission_id": {
            "$regex": course_id,
            "$options": "i"
        },
        "question_id": {
            "$ne": None
        },
    }

    print(filtered_ids)
    if filtered_ids is not None:
        submissions_query["course_learner_id"] = {
            "$in": filtered_ids
        }

    submissions = db["submissions"].find(submissions_query)
    print(submissions)
    submissions_df = pd.DataFrame(submissions)
    if submissions_df.empty:
        print(
            f"No submissions found for course {course_id}")
        return pd.DataFrame()

    submissions_df = submissions_df[~submissions_df['submission_id'].str.contains(
        "undefined")]

    submissions_df['submission_timestamp'] = pd.to_datetime(
        submissions_df['submission_timestamp'])

    submissions_df = submissions_df.drop(columns=['_id'])
    return submissions_df


def get_chapter_start_time_by_video_id(metadata_df: pd.DataFrame, video_id: str) -> pd.Timestamp:
    child_parent_map = metadata_df["object"]["child_parent_map"]
    matching_keys = [key for key in child_parent_map if video_id in key]
    block_id = metadata_df["object"]["child_parent_map"][matching_keys[0]]
    section_id = child_parent_map[block_id]
    chapter_id = child_parent_map[section_id]
    return pd.Timestamp(metadata_df["object"]["element_time_map"][chapter_id])


def get_video_interactions(db: Database, course_id: str, filtered_ids: list[str]) -> pd.DataFrame:
    video_interactions_query = construct_query(
        "course_learner_id", course_id, filtered_ids)
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

    video_interactions_df = video_interactions_df.drop(columns=['_id', 'type', 'watch_duration', 'times_forward_seek', 'duration_forward_seek',
                                                       'times_backward_seek', 'duration_backward_seek', 'times_speed_up', 'times_speed_down', 'times_pause', 'duration_pause'])
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
    course_learner_query = construct_query("course_learner_id", course_id)
    course_learners = db["course_learner"].find(course_learner_query)
    course_learners_df = pd.DataFrame(course_learners)

    if course_learners_df.empty:
        raise ValueError(f"No course learners found for course {course_id}")

    course_learners_df = course_learners_df.drop(
        columns=['_id', 'final_grade', 'enrollment_mode', 'certificate_status', 'register_time', 'group_type', 'group_name', 'segment'])
    return course_learners_df


def get_ora_sessions(db: Database, course_id: str, filtered_ids: list[str]) -> pd.DataFrame:
    ora_sessions_query = construct_query(
        "course_learner_id", course_id, filtered_ids)
    ora_sessions = db["ora_sessions"].find(ora_sessions_query)
    ora_sessions_df = pd.DataFrame(ora_sessions)

    if ora_sessions_df.empty:
        print(
            f"No ORA sessions found for course {course_id}")
        ora_sessions_df['start_time'] = None
        ora_sessions_df['end_time'] = None
        ora_sessions_df['sessionId'] = None
        ora_sessions_df['course_learner_id'] = None

        return ora_sessions_df

    ora_sessions_df['start_time'] = pd.to_datetime(
        ora_sessions_df['start_time'])
    ora_sessions_df['end_time'] = pd.to_datetime(
        ora_sessions_df['end_time'])

    ora_sessions_df = ora_sessions_df.drop(
        columns=['_id', 'times_save', 'times_peer_assess', 'submitted', 'self_assessed', 'assessment_id'])
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


def engagement_by_submissions(submissions_by_learner: pd.DataFrame, due_dates_per_assessment_period: pd.Series, assessment_periods: list[pd.Interval], learner_id: str) -> Tuple[str]:
    submissions_for_learner = submissions_by_learner[
        submissions_by_learner['course_learner_id'] == learner_id]

    elements_completed_on_time = set()
    elements_completed_late = set()
    engagement = pd.Series()

    for assessment_period in assessment_periods:
        elements_due_in_period = set(
            due_dates_per_assessment_period.get(assessment_period, []))
        # If there are no submissions due, it should not count for the learner's engagement
        if not any(["problem+block@" in element for element in elements_due_in_period]):
            engagement[assessment_period] = ""
            continue
        period_submissions = get_learner_submissions_per_assessment_period(
            submissions_for_learner, assessment_period)
        for element in elements_due_in_period:
            element_in_period_submissions = any(
                period_submissions['question_id'].str.contains(element))
            element_in_all_submissions = any(
                submissions_for_learner['question_id'].str.contains(element))

            if element_in_period_submissions:
                elements_completed_on_time.add(element)
            elif element_in_all_submissions:
                elements_completed_late.add(element)

        if elements_due_in_period.issubset(elements_completed_on_time):
            engagement[assessment_period] = "T"
        elif elements_due_in_period.issubset(elements_completed_on_time.union(elements_completed_late)):
            engagement[assessment_period] = "B"
        elif not elements_due_in_period.isdisjoint(elements_completed_on_time.union(elements_completed_late)):
            engagement[assessment_period] = "A"
        else:
            engagement[assessment_period] = "O"
    engagement = engagement.sort_index()
    return tuple(engagement.values)


def engagement_by_video_interactions(video_interactions: pd.DataFrame, due_dates_per_assessment_period: pd.Series, metadata_df: pd.DataFrame, assessment_periods: list[pd.Interval], learner_id: str) -> pd.Series:
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

    for assessment_period in assessment_periods:
        if assessment_period in has_watched_video_in_assessment_period:
            engagement[assessment_period] = "A"
        else:
            engagement[assessment_period] = ""

    engagement = engagement.sort_index()
    return tuple(engagement.values)


def engagement_by_quiz_sessions(quiz_sessions: pd.DataFrame, quizzes_due_per_assessment_period: pd.Series, assessment_periods: list[pd.Interval], learner_id: str) -> Tuple[str]:
    if learner_id != "course-v1:DelftX+EX101x+3T2015_4f51ab1f44e5a496e86dfff703efad2d":
        return tuple()

    quiz_sessions_for_learner = quiz_sessions[quiz_sessions['course_learner_id'] == learner_id]
    assessment_periods_with_all_quizzes_on_time = set()
    assessment_periods_with_all_quizzes_late_or_on_time = set()
    assessment_periods_with_some_quizzes_done = set()

    engagement = pd.Series()

    for assessment_period, elements_due in quizzes_due_per_assessment_period.items():
        if not elements_due:
            engagement[assessment_period] = ""
            continue

        quiz_sessions_in_period = get_learner_quiz_sessions_per_assessment_period(
            quiz_sessions_for_learner, assessment_period)
        quizzes_done_on_time = 0
        quizzes_done = 0
        for element in elements_due:
            element_in_period_quiz_sessions = any(
                quiz_sessions_in_period['sessionId'].str.contains(element))
            if element_in_period_quiz_sessions:
                quizzes_done_on_time += 1

            element_in_all_quiz_sessions = any(
                quiz_sessions_for_learner['sessionId'].str.contains(element))
            # Find if there is any quiz sesssion for the element with an end time before
            if element_in_all_quiz_sessions:
                quizzes_done += 1
        if quizzes_done_on_time == len(elements_due):
            assessment_periods_with_all_quizzes_on_time.add(assessment_period)
        elif quizzes_done_on_time + quizzes_done == len(elements_due):
            assessment_periods_with_all_quizzes_late_or_on_time.add(
                assessment_period)
        elif quizzes_done_on_time + quizzes_done > 0:
            assessment_periods_with_some_quizzes_done.add(assessment_period)

    for assessment_period in assessment_periods:
        if assessment_period in assessment_periods_with_all_quizzes_on_time:
            engagement[assessment_period] = "T"
        elif assessment_period in assessment_periods_with_all_quizzes_late_or_on_time:
            engagement[assessment_period] = "B"
        elif assessment_period in assessment_periods_with_some_quizzes_done:
            engagement[assessment_period] = "A"
        elif quizzes_due_per_assessment_period.get(assessment_period, None) is not None:
            engagement[assessment_period] = "O"
        # Should never be reached
        else:
            raise ValueError("Error in quiz engagement calculation")
    engagement = engagement.sort_index()
    return tuple(engagement.values)


def engagement_by_ora_sessions(ora_sessions: pd.DataFrame, due_dates_per_assessment_period: pd.Series, assessment_periods: list[pd.Interval], learner_id: str) -> Tuple[str]:
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
                ora_sessions_for_learner['sessionId'].str.contains(element))
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

    for assessment_period in assessment_periods:
        if assessment_period in assessment_periods_with_all_oras_on_time:
            engagement[assessment_period] = "T"
        elif assessment_period in assessment_periods_with_all_oras_late_or_on_time:
            engagement[assessment_period] = "B"
        elif assessment_period in assessment_periods_with_some_oras_done:
            engagement[assessment_period] = "A"
        elif ora_due_dates_per_assessment_period.get(assessment_period, None) is not None:
            engagement[assessment_period] = "O"
        else:
            engagement[assessment_period] = ""

    engagement = engagement.sort_index()
    return tuple(engagement.values)


def construct_learner_engagement_mapping(quiz_sessions_df: pd.DataFrame, metadata_df: pd.DataFrame, course_learner_df: pd.DataFrame, video_interactions_df: pd.DataFrame, submissions_df: pd.DataFrame, ora_sessions_df: pd.DataFrame, learner_demographic_df: pd.DataFrame) -> Tuple[Counter[tuple, int], Counter[tuple, int]]:
    course_start_date = pd.Timestamp(metadata_df["object"]["start_date"])
    due_dates = {key: pd.Timestamp(
        value) for key, value in metadata_df["object"]["element_time_map_due"].items()}
    child_parent_map = metadata_df["object"]["child_parent_map"]

    # valid_course_learner_ids = set(course_learner_df['course_learner_id'])
    # video_interactions_df = video_interactions_df[video_interactions_df['course_learner_id'].isin(
    #     valid_course_learner_ids)]
    # quiz_sessions_df = quiz_sessions_df[quiz_sessions_df['course_learner_id'].isin(
    #     valid_course_learner_ids)]
    # ora_sessions_df = ora_sessions_df[ora_sessions_df['course_learner_id'].isin(
    #     valid_course_learner_ids)]
    # submissions_df = submissions_df[submissions_df['course_learner_id'].isin(
    #     valid_course_learner_ids)]

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

    assessment_periods = due_dates_per_assessment_period.index
    tqdm.pandas()

    course_learner_df['quiz_engagement'] = course_learner_df['course_learner_id'].progress_apply(
        lambda learner_id: engagement_by_quiz_sessions(quiz_sessions_df, quizzes_due_per_assessment_period, assessment_periods, learner_id))
    print(course_learner_df.head())
    course_learner_df['video_engagement'] = course_learner_df['course_learner_id'].progress_apply(
        lambda learner_id: engagement_by_video_interactions(video_interactions_df, due_dates_per_assessment_period, metadata_df, assessment_periods, learner_id))
    print(course_learner_df.head())

    course_learner_df['submission_engagement'] = course_learner_df['course_learner_id'].progress_apply(
        lambda learner_id: engagement_by_submissions(submissions_df, submissions_due_per_assessment_period, assessment_periods, learner_id))
    print(course_learner_df.head())

    course_learner_df['ora_engagement'] = course_learner_df['course_learner_id'].progress_apply(
        lambda learner_id: engagement_by_ora_sessions(ora_sessions_df, ora_due_per_assessment_period, assessment_periods, learner_id))
    print(course_learner_df.head())

    course_learner_df['engagement'] = course_learner_df.progress_apply(
        lambda row: process_engagement(row['quiz_engagement'], row['video_engagement'], row['submission_engagement'], row['ora_engagement'], due_dates_per_assessment_period
                                       ), axis=1)
    print(course_learner_df.head())

    male_counts = course_learner_df[course_learner_df['gender']
                                    == 'm']['engagement'].value_counts()
    print(male_counts)
    female_counts = course_learner_df[course_learner_df['gender']
                                      == 'f']['engagement'].value_counts()

    return male_counts, female_counts


def process_engagement(quiz_engagement: tuple[str], video_engagement: tuple[str], submission_engagement: tuple[str], ora_engagement: tuple[str], assessment_periods: tuple[str]) -> tuple[str]:
    engagement = tuple()
    assert len(quiz_engagement) == len(video_engagement) == len(
        submission_engagement) == len(ora_engagement) == len(assessment_periods)

    for idx, assessment_periods in enumerate(assessment_periods):
        quiz_engagement_for_period = quiz_engagement[idx]
        video_engagement_for_period = video_engagement[idx]
        submission_engagement_for_period = submission_engagement[idx]
        ora_engagement_for_period = ora_engagement[idx]

        non_video_statuses = set([quiz_engagement_for_period,
                                  submission_engagement_for_period, ora_engagement_for_period])

        non_video_statuses.discard("")

        # Rule 1: If any status is 'A', return 'A'
        if "A" in non_video_statuses:
            engagement += ("A",)
        # Rule 2: If it's a mix of 'T' and 'B', or just 'B', return 'B'
        elif "B" in non_video_statuses and non_video_statuses == {"T", "B"}:
            engagement += ("B",)
        # Rule 3: If all are 'T', return 'T'
        elif non_video_statuses == {"T"}:
            engagement += ("T",)
        # Rule 4: If all non-video engagements are "", check video engagement
        elif not non_video_statuses:
            # If video_engagement is 'A', return 'A'. Otherwise, return 'O' since video_engagement can only be 'A' or 'O'.
            engagement += ("A" if video_engagement_for_period == "A" else "O",)
        # Default case to catch any unforeseen combinations, primarily for safety
        engagement += ("O",)

    return engagement


def get_learner_quiz_sessions_per_assessment_period(learner_quiz_sessions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    if not learner_quiz_sessions.empty:
        print(learner_quiz_sessions)
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


def engagement_to_numeric(engagement: pd.Series) -> list:
    label_to_numeric = {'T': 3, 'B': 2, 'A': 1, 'O': 0}
    return [label_to_numeric[label] for label in engagement]


def calculate_k_means_clusters(learner_engagement_mapping: pd.Series, title: str) -> None:
    engagement_numeric = [engagement_to_numeric(
        pattern) for pattern in learner_engagement_mapping.keys()]
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


def get_quiz_sessions(db: Database, course_id: str, filtered_ids: list[str]) -> pd.DataFrame:
    quiz_sessions_query = construct_query("course_id", course_id, filtered_ids)
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
    db = client["edx-test"]

    courses = get_courses(db)

    for _, course in courses.iterrows():
        try:
            full_course_id = course["course_id"]
            if "DelftX+EX101x+3T2015" not in full_course_id:
                continue
            mooc_id = full_course_id.split("+")[-2]
            course_run = full_course_id.split("+")[-1]
            course_id_with_course_run = f"{mooc_id}_{course_run}"
            pattern = r"(\w+_\d+T)_(\d{4})"
            if re.match(pattern, course_id_with_course_run):
                course_id_with_course_run = re.sub(
                    pattern, r"\1\2", course_id_with_course_run)
            escaped_course_id = escape_id(full_course_id)

            metadata = get_metadata(db, course_id_with_course_run)
            # course_id quiz sessions: "course-v1:DelftX+EX101x+2T2018"
            # metadata: EX101x_3T2015_run2
            if not has_due_dates(metadata):
                # raise ValueError("Course does not have due dates.")
                continue
            course_learner = get_course_learner(
                db, escaped_course_id)
            course_learner = course_learner.merge(get_learner_demographic(
                db, escaped_course_id)[['course_learner_id', 'gender']], on='course_learner_id', how='left')
            filtered_learners = course_learner[course_learner['gender'].isin([
                'm', 'f'])]

            filtered_ids: list[str] = filtered_learners['course_learner_id'].unique(
            ).tolist()

            submissions = get_submissions(db, escaped_course_id, filtered_ids)
            print("Submissions done")
            video_interactions = get_video_interactions(
                db, escaped_course_id, filtered_ids)
            print("Video interactions done")
            quiz_sessions = get_quiz_sessions(
                db, "course-v1:DelftX+" + "+".join(course_id_with_course_run.split("_")[:2]), filtered_ids)
            print("Quiz sessions done")
            ora_sessions = get_ora_sessions(
                db, escaped_course_id, filtered_ids)

            # filter course_learner to the first 500
            course_learner = course_learner.head(500)

            print("All data loaded")
            male_learner_engagement_mapping, female_learner_engagement_mapping = construct_learner_engagement_mapping(quiz_sessions, metadata,
                                                                                                                      course_learner, video_interactions, submissions, ora_sessions, filtered_learners)
            calculate_k_means_clusters(
                male_learner_engagement_mapping, full_course_id)
            calculate_k_means_clusters(
                female_learner_engagement_mapping, full_course_id)

        except Exception as ex:
            print(''.join(traceback.format_exception(type(ex),
                                                     value=ex, tb=ex.__traceback__)))


if __name__ == '__main__':
    main()
