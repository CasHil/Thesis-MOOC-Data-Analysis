import re
from typing import Tuple
import traceback
from collections import Counter
from functools import partial

import pandas as pd
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

    return pd.DataFrame(learner_demographic)


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
    return video_interactions_df


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

    return ora_sessions_df


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


def get_gender_by_course_learner_id(learner_demographic: pd.DataFrame, course_learner_id: str) -> str:
    user = learner_demographic[learner_demographic['course_learner_id']
                               == course_learner_id]
    if user.empty:
        return ""
    return user.iloc[0]["gender"]


def process_learner_row(row: pd.Series, submissions_by_learner: pd.DataFrame, videos_by_learner: pd.DataFrame, quizzes_by_learner: pd.DataFrame, oras_by_learner: pd.DataFrame, due_dates_per_assessment_period: pd.Series, flattened_due_dates_series: pd.Series, metadata_df: pd.DataFrame) -> tuple[str, ...]:

    # https://dl.acm.org/doi/abs/10.1145/2460296.2460330
    # On track if all elements with due dates have been completed or have a session with an end date before the due date.
    on_track = "T"
    # Behind if all elements have been completed, but there is at least one after the due date.
    behind = "B"
    # Auditing if some elements have been completed, or there are video interactions or quiz sessions.
    auditing = "A"
    # Out if no elements have been completed, and there are no video interactions or quiz sessions.
    out = "O"

    course_learner_id = row["course_learner_id"]

    submissions = submissions_by_learner[submissions_by_learner['course_learner_id']
                                         == course_learner_id]
    video_interactions = videos_by_learner[videos_by_learner['course_learner_id']
                                           == course_learner_id]
    quiz_sessions = quizzes_by_learner[quizzes_by_learner['course_learner_id']
                                       == course_learner_id]
    ora_sessions = oras_by_learner[oras_by_learner['course_learner_id']
                                   == course_learner_id]

    has_submissions = course_learner_id in submissions['course_learner_id'].values
    has_video_interactions = course_learner_id in video_interactions['course_learner_id'].values
    has_quiz_sessions = course_learner_id in quiz_sessions['course_learner_id'].values
    has_ora_sessions = course_learner_id in ora_sessions['course_learner_id'].values

    elements_completed_on_time = set()
    elements_completed_late = set()
    has_watched_video_in_assessment_period = set()

    learner_engagement = pd.Series([None for _ in range(
        len(due_dates_per_assessment_period))])

    if not has_submissions and not has_video_interactions and not has_quiz_sessions and not has_ora_sessions:
        out_learning_trajectories = tuple(
            [out for _ in range(len(due_dates_per_assessment_period))])
        return out_learning_trajectories

    if has_submissions or has_ora_sessions or has_quiz_sessions:
        for assessment_period, elements_due in due_dates_per_assessment_period.items():

            assessment_period_submissions = get_learner_submissions_per_assessment_period(
                submissions_by_learner, assessment_period)
            assessment_period_quiz_sessions: pd.Series = get_learner_quiz_sessions_per_assessment_period(
                quizzes_by_learner, assessment_period)
            assessment_period_ora_sessions = get_learner_ora_sessions_per_assessment_period(
                oras_by_learner, assessment_period)

            for element in elements_due:
                element_in_week_submissions = any(
                    assessment_period_submissions['question_id'].str.contains(element))
                element_in_week_ora_sessions = any(
                    assessment_period_ora_sessions['assessment_id'] == element)
                element_in_week_quiz_sessions = any(
                    assessment_period_quiz_sessions['sessionId'] == element)
                if element_in_week_submissions or element_in_week_ora_sessions or element_in_week_quiz_sessions:
                    elements_completed_on_time.add(element)
                    continue

                element_in_all_submissions = any(
                    submissions_by_learner['question_id'].str.contains(element))
                element_in_all_ora_sessions = any(
                    oras_by_learner['assessment_id'] == element)
                element_in_all_quiz_sessions = any(
                    quizzes_by_learner['sessionId'] == element)

                if element_in_all_submissions or element_in_all_ora_sessions or element_in_all_quiz_sessions:
                    if flattened_due_dates_series.get(element, pd.Timestamp.max) < assessment_period.right:
                        elements_completed_on_time.add(element)
                    else:
                        elements_completed_late.add(element)

    if has_video_interactions:
        video_interactions = video_interactions.copy()
        video_interactions['chapter_start_time'] = video_interactions['video_id'].apply(
            lambda x: get_chapter_start_time_by_video_id(metadata_df, x))
        chapter_to_assessment_period = {
            period.left: period for period in due_dates_per_assessment_period.index}

        video_interactions['assessment_period'] = video_interactions['chapter_start_time'].map(
            chapter_to_assessment_period)
        valid_video_interactions = video_interactions.dropna(
            subset=['assessment_period'])
        has_watched_video_in_assessment_period.update(
            valid_video_interactions['assessment_period'].unique())

    all_completed_elements = elements_completed_on_time.union(
        elements_completed_late)

    for idx, assessment_period in enumerate(due_dates_per_assessment_period.index):
        period_elements = set(
            due_dates_per_assessment_period[assessment_period])
        if period_elements.issubset(elements_completed_on_time):
            learner_engagement[idx] = on_track
        elif period_elements.issubset(all_completed_elements):
            learner_engagement[idx] = behind
        elif not period_elements.isdisjoint(all_completed_elements) or assessment_period in has_watched_video_in_assessment_period:
            learner_engagement[idx] = auditing
        else:
            learner_engagement[idx] = out

    return tuple(learner_engagement)


def construct_learner_engagement_mapping(quiz_sessions_df: pd.DataFrame, metadata_df: pd.DataFrame, course_learner_df: pd.DataFrame, video_interactions_df: pd.DataFrame, submissions_df: pd.DataFrame, ora_sessions_df: pd.DataFrame, learner_demographic_df: pd.DataFrame) -> Tuple[Counter[tuple, int], Counter[tuple, int]]:
    course_start_date = pd.Timestamp(metadata_df["object"]["start_date"])
    due_dates = {key: pd.Timestamp(
        value) for key, value in metadata_df["object"]["element_time_map_due"].items()}
    child_parent_map = metadata_df["object"]["child_parent_map"]

    for element in list(due_dates):
        if "sequential" in element:
            element_children = {
                key: due_dates[element] for key in child_parent_map if child_parent_map[key] == element}
            del due_dates[element]
            due_dates.update(element_children)

    course_learner_df['gender'] = course_learner_df['course_learner_id'].apply(
        lambda id: get_gender_by_course_learner_id(learner_demographic_df, id))

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

    flattened_due_dates_series = pd.Series(flattened_due_dates)

    due_dates_per_assessment_period: pd.Series = get_due_dates_per_assessment_period(
        flattened_due_dates, course_start_date)

    tqdm.pandas(total=len(course_learner_df),
                desc="Processing learners")
    process_learner_row_partial = partial(process_learner_row, submissions_by_learner=submissions_df, videos_by_learner=video_interactions_df, quizzes_by_learner=quiz_sessions_df,
                                          oras_by_learner=ora_sessions_df, due_dates_per_assessment_period=due_dates_per_assessment_period, flattened_due_dates_series=flattened_due_dates_series, metadata_df=metadata_df)

    course_learner_df["engagement_summary"] = course_learner_df.progress_apply(
        process_learner_row_partial, axis=1)

    engagement_counts = course_learner_df.groupby(
        ['gender', 'engagement_summary']).size().reset_index(name='count')

    men_counts_df = engagement_counts[engagement_counts['gender'] == 'm']
    women_counts_df = engagement_counts[engagement_counts['gender'] == 'f']

    men_engagement_counter = Counter(
        dict(zip(men_counts_df['engagement_summary'], men_counts_df['count'])))
    women_engagement_counter = Counter(
        dict(zip(women_counts_df['engagement_summary'], women_counts_df['count'])))
    return men_engagement_counter, women_engagement_counter


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
    return quiz_sessions_df


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

            learner_demographic = get_learner_demographic(
                db, escaped_course_id)
            print("Learner demographic done")
            submissions = get_submissions(db, escaped_course_id)
            print("Submissions done")
            video_interactions = get_video_interactions(db, escaped_course_id)
            print("Video interactions done")
            quiz_sessions = get_quiz_sessions(db, escaped_course_id)
            print("Quiz sessions done")
            course_learner = get_course_learner(db, escaped_course_id)
            ora_sessions = get_ora_sessions(db, escaped_course_id)

            filtered_learners = learner_demographic[learner_demographic['gender'].isin([
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
