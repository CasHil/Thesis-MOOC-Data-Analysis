import re
from typing import Tuple
import traceback
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pymongo import MongoClient
from pymongo.database import Database
from tqdm import tqdm
import matplotlib.pyplot as plt


def find_videos_per_assessment_period(df: pd.DataFrame) -> pd.DataFrame:
    df['videos_per_assessment_period'] = df['videos_watched'] / df['weeks_enrolled']
    return df


def find_assignments_per_assessment_period(df: pd.DataFrame) -> pd.DataFrame:
    df['assignments_per_assessment_period'] = df['assignments_submitted'] / \
        df['weeks_enrolled']
    return df


def get_courses(db: Database) -> pd.DataFrame:
    courses = db["courses"].find()
    courses_df = pd.DataFrame(courses)

    if courses_df.empty:
        raise ValueError(f"No courses found.")

    return courses_df


def classify_submission(row: pd.Series, element_time_map_due: dict[str, pd.Timestamp]):
    element_id = row['element_id']
    submission_timestamp = row['submission_timestamp']
    due_date = element_time_map_due.get(element_id)
    if due_date and submission_timestamp <= due_date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1):
        return 'on track'
    return 'behind'


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


def get_course_elements(db: Database, course_id: str) -> pd.DataFrame:
    course_elements_query = {
        "course_id": {
            "$regex": course_id,
            "$options": "i"
        }
    }
    course_elements = db["course_elements"].find(course_elements_query)
    course_elements_df = pd.DataFrame(course_elements)

    if course_elements_df.empty:
        raise ValueError(f"No course elements found for course {course_id}")

    return course_elements_df


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


def get_assessments(db: Database, course_id: str) -> pd.DataFrame:
    assessments_query = {
        "assessment_id": {
            "$regex": course_id,
            "$options": "i"
        }
    }
    assessments = db["assessments"].find(assessments_query)
    assessments_df = pd.DataFrame(assessments)
    if assessments_df.empty:
        print(
            f"No assessments found for course {course_id}")
        return pd.DataFrame()

    return pd.DataFrame(assessments_df)


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
        return None
    return user.iloc[0]["gender"]


def construct_learner_engagement_mapping(course_elements_df: pd.DataFrame, quiz_sessions_df: pd.DataFrame, metadata_df: pd.DataFrame, course_learner_df: pd.DataFrame, video_interactions_df: pd.DataFrame, submissions_df: pd.DataFrame, ora_sessions_df: pd.DataFrame, learner_demographic_df: pd.DataFrame) -> tuple[dict[tuple, int], dict[tuple, int]]:
    course_start_date = pd.Timestamp(metadata_df["object"]["start_date"])
    course_end_date = pd.Timestamp(metadata_df["object"]["end_date"])
    due_dates = {key: pd.Timestamp(
        value) for key, value in metadata_df["object"]["element_time_map_due"].items()}
    child_parent_map = metadata_df["object"]["child_parent_map"]

    for element in list(due_dates):
        if "sequential" in element:
            element_children = {
                key: due_dates[element] for key in child_parent_map if child_parent_map[key] == element}
            del due_dates[element]
            due_dates.update(element_children)

    course_element_ids_set = set(course_elements_df["element_id"])

    male_learning_trajectories = Counter()
    female_learning_trajectories = Counter()

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

    final_assessment_period: pd.Timestamp = due_dates_per_assessment_period.iloc[-1]

    # https://dl.acm.org/doi/abs/10.1145/2460296.2460330

    # On track if all elements with due dates have been completed or have a session with an end date before the due date.
    on_track = "T"
    # Behind if all elements have been completed, but there is at least one after the due date.
    behind = "B"
    # Auditing if some elements have been completed, or there are video interactions or quiz sessions.
    auditing = "A"
    # Out if no elements have been completed, and there are no video interactions or quiz sessions.
    out = "O"

    for index, course_learner in tqdm(course_learner_df.iterrows(), total=course_learner_df.shape[0]):
        course_learner_id = course_learner["course_learner_id"]
        gender = get_gender_by_course_learner_id(
            learner_demographic_df, course_learner_id)

        learner_submissions = submissions_df[submissions_df['course_learner_id']
                                             == course_learner_id]

        learner_submission_ids: pd.DataFrame = learner_submissions["submission_id"]
        learner_submission_element_ids: pd.DataFrame = learner_submission_ids.apply(
            process_element_id)
        learner_video_interactions = video_interactions_df[
            video_interactions_df["course_learner_id"] == course_learner_id]
        learner_quiz_sessions = quiz_sessions_df[quiz_sessions_df["course_learner_id"]
                                                 == course_learner_id]
        learner_ora_sessions = ora_sessions_df[ora_sessions_df["course_learner_id"]
                                               == course_learner_id]

        has_submissions = not submissions_df[submissions_df['course_learner_id']
                                             == course_learner_id].empty
        has_video_interactions = not video_interactions_df[
            video_interactions_df['course_learner_id'] == course_learner_id].empty
        has_quiz_sessions = not quiz_sessions_df[quiz_sessions_df['course_learner_id']
                                                 == course_learner_id].empty
        has_ora_sessions = not ora_sessions_df[ora_sessions_df['course_learner_id']
                                               == course_learner_id].empty

        if not has_submissions and not has_video_interactions and not has_quiz_sessions and not has_ora_sessions:
            out_learning_trajectories = tuple(
                [out for _ in range(len(due_dates_per_assessment_period))])
            if gender == "m":
                male_learning_trajectories[out_learning_trajectories] += 1
            elif gender == "f":
                female_learning_trajectories[out_learning_trajectories] += 1
            continue

        # Elements in that have been completed on time
        elements_completed_on_time = set()

        # Elements that have been completed after the due date of the assessment period
        elements_completed_late = set()

        # Video interactions per assessment period. Does not have to be by the assessment period's due date.
        has_watched_video_in_assessment_period = set()

        learner_engagement = pd.Series([None for _ in range(
            len(due_dates_per_assessment_period))])

        if has_submissions or has_ora_sessions or has_quiz_sessions:
            for assessment_period, elements_due in due_dates_per_assessment_period.items():

                week_submissions = get_learner_submissions_per_assessment_period(
                    learner_submissions, assessment_period)
                week_quiz_sessions: pd.Series = get_learner_quiz_sessions_per_assessment_period(
                    learner_quiz_sessions, assessment_period)
                week_ora_sessions = get_learner_ora_sessions_per_assessment_period(
                    learner_ora_sessions, assessment_period)

                for element in elements_due:
                    element_in_week_submissions = any(
                        week_submissions['question_id'].str.contains(element))
                    element_in_week_ora_sessions = any(
                        week_ora_sessions['assessment_id'] == element)
                    element_in_week_quiz_sessions = any(
                        week_quiz_sessions['sessionId'] == element)
                    if element_in_week_submissions or element_in_week_ora_sessions or element_in_week_quiz_sessions:
                        elements_completed_on_time.add(element)
                        continue

                    element_in_all_submissions = any(
                        learner_submissions['question_id'].str.contains(element))
                    element_in_all_ora_sessions = any(
                        learner_ora_sessions['assessment_id'] == element)
                    element_in_all_quiz_sessions = any(
                        learner_quiz_sessions['sessionId'] == element)

                    if element_in_all_submissions or element_in_all_ora_sessions or element_in_all_quiz_sessions:
                        if flattened_due_dates_series.get(element, pd.Timestamp.max) < assessment_period.right:
                            elements_completed_on_time.add(element)
                        else:
                            elements_completed_late.add(element)

        if has_video_interactions:
            learner_video_interactions = learner_video_interactions.copy()
            learner_video_interactions['chapter_start_time'] = learner_video_interactions['video_id'].apply(
                lambda x: get_chapter_start_time_by_video_id(metadata_df, x))
            chapter_to_assessment_period = {
                period.left: period for period in due_dates_per_assessment_period.index}

            learner_video_interactions['assessment_period'] = learner_video_interactions['chapter_start_time'].map(
                chapter_to_assessment_period)
            valid_video_interactions = learner_video_interactions.dropna(
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

        learner_engagement_tuple = tuple(learner_engagement)

        if gender == "m":
            male_learning_trajectories[learner_engagement_tuple] += 1
        elif gender == "f":
            female_learning_trajectories[learner_engagement_tuple] += 1

    return male_learning_trajectories, female_learning_trajectories


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

    temp_df.sort_values(by='due_date', inplace=True)

    grouped = temp_df.groupby('due_date')['block_id'].apply(list)

    grouped = grouped.sort_index()
    periods_with_due_dates = pd.Series(index=pd.IntervalIndex.from_arrays(
        [course_start_date] + grouped.index[:-1].tolist(), grouped.index.tolist()), data=grouped.values)

    return periods_with_due_dates


def engagement_to_numeric(engagement: tuple) -> list:
    label_to_numeric = {'T': 3, 'B': 2, 'A': 1, 'O': 0}
    return [label_to_numeric[label] for label in engagement]


def calculate_k_means_clusters(learner_engagement_mapping: dict[tuple, int], title: str) -> None:
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

            course_elements = get_course_elements(db, escaped_course_id)
            print("Course elements done")
            learner_demographic = get_learner_demographic(
                db, escaped_course_id)
            print("Learner demographic done")
            assessments = get_assessments(db, escaped_course_id)
            print("Assessments done")
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
            male_learner_engagement_mapping, female_learner_engagement_mapping = construct_learner_engagement_mapping(course_elements, quiz_sessions, metadata,
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
