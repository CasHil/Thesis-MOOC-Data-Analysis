import re
from typing import Tuple
import traceback
from collections import Counter
from functools import partial

import pandas as pd
import dask.dataframe as dd
from dask.dataframe import DataFrame as DaskDataFrame
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from pymongo import MongoClient
from pymongo.database import Database
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


def get_courses(db: Database) -> pd.DataFrame:
    courses = db["courses"].find()

    courses_df = pd.DataFrame(courses)

    if courses_df.empty:
        raise ValueError("No courses found.")

    courses_df['start_time'] = pd.to_datetime(courses_df['start_time'])
    courses_df['end_time'] = pd.to_datetime(courses_df['end_time'])

    return courses_df


def construct_query(field: str, search_string: str, filtered_ids: set[str] = None, exact = False):
    if exact:
        query = {
            field: search_string
        }
    else:
        query = {
            field: {
                "$regex": search_string,
                "$options": "i"
            }
        }

    if filtered_ids is not None:
        query["course_learner_id"] = {
            "$in": list(filtered_ids)
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


def get_learner_demographic(db: Database, course_id: str, filtered_ids: set[str]) -> pd.DataFrame:
    learner_demographic_query = construct_query(
        "course_id", course_id, filtered_ids, exact=True)
    learner_demographic = db["learner_demographic"].find(
        learner_demographic_query)

    learner_demographic_df = pd.DataFrame(learner_demographic)

    if learner_demographic_df.empty:
        raise ValueError(
            f"No learner demographic data found for course {course_id}")

    return downcast_numeric_columns(learner_demographic_df)


def process_element_id(x: str) -> str:
    return "+type@".join(x.split("+type@")[1:])


def escape_id(course_id: str) -> str:
    return re.escape(course_id)


def get_submissions(db: Database, course_id: str, filtered_ids: set[str]) -> pd.DataFrame:
    submissions_query = {
        "course_id": course_id,
        "question_id": {
            "$ne": None
        },
    }

    if filtered_ids is not None:
        submissions_query["course_learner_id"] = {
            "$in": list(filtered_ids)
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

    submissions_df = submissions_df.drop(columns=['_id'])
    return submissions_df


def get_chapter_start_time_by_video_id(metadata_df: pd.DataFrame, video_id: str) -> pd.Timestamp:
    child_parent_map = metadata_df["object"]["child_parent_map"]
    matching_keys = [key for key in child_parent_map if video_id in key]
    block_id = metadata_df["object"]["child_parent_map"][matching_keys[0]]
    section_id = child_parent_map[block_id]
    chapter_id = child_parent_map[section_id]
    return pd.Timestamp(metadata_df["object"]["element_time_map"][chapter_id])


def get_video_interactions(db: Database, course_id: str, filtered_ids: set[str]) -> pd.DataFrame:
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


def get_ora_sessions(db: Database, course_id: str, filtered_ids: set[str]) -> pd.DataFrame:
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


def get_learner_submissions_in_assessment_period(learner_submissions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    learner_submissions_before_due_date = learner_submissions[learner_submissions['submission_timestamp'] < assessment_period.right]
    return learner_submissions_before_due_date



def calculate_submission_engagement(learner_submissions: pd.DataFrame, due_dates_per_assessment_period: pd.Series, assessment_periods: list[pd.Interval], learner_id: str, course_has_due_dates: bool) -> Tuple[str]:

    engagement = pd.Series(index=pd.Index(assessment_periods), dtype='object').fillna('')

    if course_has_due_dates:
        elements_completed_on_time = set()
        elements_completed_late = set()

        for assessment_period in assessment_periods:
            elements_due_in_period = set(due_dates_per_assessment_period.get(assessment_period, []))
            period_submissions = learner_submissions[
                learner_submissions['question_id'].isin(elements_due_in_period) &
                (pd.to_datetime(learner_submissions['submission_time']) <= assessment_period.right)
            ]

            elements_completed_on_time.update(period_submissions['question_id'].unique())

            late_submissions = learner_submissions[
                learner_submissions['question_id'].isin(elements_due_in_period) &
                (pd.to_datetime(learner_submissions['submission_time']) > assessment_period.right)
            ]

            elements_completed_late.update(late_submissions['question_id'].unique())

        for period in assessment_periods:
            due_elements = set(due_dates_per_assessment_period.get(period, []))
            completed_on_time = due_elements.intersection(elements_completed_on_time)
            completed_late = due_elements.intersection(elements_completed_late)
            all_completed = completed_on_time.union(completed_late)

            if due_elements.issubset(completed_on_time):
                engagement[period] = "T"
            elif due_elements.issubset(all_completed):
                engagement[period] = "B"
            elif not all_completed.isdisjoint(due_elements):
                engagement[period] = "A"
            else:
                engagement[period] = "O"

    else:
        for assessment_period in assessment_periods:
            elements_due_in_period = set(due_dates_per_assessment_period.get(assessment_period, []))
            period_submissions = learner_submissions[
                learner_submissions['question_id'].isin(elements_due_in_period)
            ]

            submissions_done = len(period_submissions)

            if submissions_done == len(elements_due_in_period):
                engagement[assessment_period] = "T"
            elif submissions_done > 0:
                engagement[assessment_period] = "A"
            elif elements_due_in_period:
                engagement[assessment_period] = "O"
            else:
                engagement[assessment_period] = ""

        for period in assessment_periods:
            due_elements = set(due_dates_per_assessment_period.get(period, []))
            completed_on_time = due_elements.intersection(elements_completed_on_time)
            completed_late = due_elements.intersection(elements_completed_late)
            all_completed = completed_on_time.union(completed_late)

            if due_elements.issubset(completed_on_time):
                engagement[period] = "T"
            elif not all_completed.isdisjoint(due_elements):
                engagement[period] = "A"
            else:
                engagement[period] = "O"

    engagement = engagement.sort_index()
    return tuple(engagement.values)


def find_period(timestamp, periods):
    for period in periods:
        if timestamp in period:
            return period
    if timestamp > periods[-1].right:
        return periods[-1]
    elif timestamp < periods[0].left:
        return periods[0]

def calculate_ora_engagement(learner_ora_sessions: pd.DataFrame, due_dates_per_assessment_period: pd.Series, assessment_periods: list[pd.Interval], course_has_due_dates: bool) -> Tuple[str]:
    engagement = pd.Series(index=pd.Index(assessment_periods), dtype='object').fillna('')

    ora_due_dates_per_assessment_period = due_dates_per_assessment_period.apply(
        lambda elements: [element for element in elements if "openassessment" in element])

    assessment_periods_with_all_oras_on_time = set()
    assessment_periods_with_all_oras_late_or_on_time = set()
    assessment_periods_with_some_oras_done = set()

    if course_has_due_dates:
        for assessment_period, elements_due in ora_due_dates_per_assessment_period.items():
            ora_sessions_in_period = get_learner_ora_sessions_per_assessment_period(
                learner_ora_sessions, assessment_period)
            
            if (ora_sessions_in_period['block_id'].isin(elements_due)).all():
                assessment_periods_with_all_oras_on_time.add(assessment_period)
            elif (ora_sessions_in_period['block_id'].isin(elements_due)).any():
                assessment_periods_with_some_oras_done.add(assessment_period)
            else:
                assessment_periods_with_all_oras_late_or_on_time.add(assessment_period)

        for period in assessment_periods:
            if period in assessment_periods_with_all_oras_on_time:
                engagement[period] = "T"
            elif period in assessment_periods_with_all_oras_late_or_on_time:
                engagement[period] = "B"
            elif period in assessment_periods_with_some_oras_done:
                engagement[period] = "A"
            elif period in ora_due_dates_per_assessment_period:
                engagement[period] = "O"
            else:
                engagement[period] = ""

    else:
        for assessment_period, elements_due in ora_due_dates_per_assessment_period.items():
            ora_sessions_in_period = get_learner_ora_sessions_per_assessment_period(
                learner_ora_sessions, assessment_period)

            if (ora_sessions_in_period['block_id'].isin(elements_due)).any():
                assessment_periods_with_some_oras_done.add(assessment_period)
            else:
                assessment_periods_with_all_oras_late_or_on_time.add(assessment_period)

        for period in assessment_periods:
            if period in assessment_periods_with_some_oras_done:
                engagement[period] = "A"
            elif period in ora_due_dates_per_assessment_period:
                engagement[period] = "O"
            else:
                engagement[period] = ""

    return tuple(engagement.values)


def get_week_of_course_element(db: Database, element_id: str) -> int:
    print(element_id)
    return db["course_elements"].find_one({"element_id": element_id})["week"]

def calculate_video_engagement(learner_video_interactions: pd.DataFrame, metadata_df: pd.DataFrame, assessment_periods: list[pd.Interval]) -> Tuple[str]:
    """
    Calculate the engagement based on pre-filtered video interactions for a single learner.
    """
    engagement = pd.Series(index=pd.Index(assessment_periods), dtype='object').fillna('')

    learner_video_interactions['chapter_start_time'] = learner_video_interactions['video_id'].apply(
        lambda vid: get_chapter_start_time_by_video_id(metadata_df, vid)
    )
    
    videos_grouped_by_period = learner_video_interactions.groupby(
        lambda idx: find_period(learner_video_interactions.loc[idx, 'chapter_start_time'], assessment_periods)
    )

    for period, videos_in_period in videos_grouped_by_period:
        if not videos_in_period.empty:
            engagement[period] = "A"

    return tuple(engagement.values)


def calculate_quiz_engagement(learner_sessions: pd.DataFrame, quizzes_due_per_assessment_period: pd.Series, assessment_periods: pd.Index | pd.MultiIndex, course_has_due_dates: bool):
    """
    Calculate the engagement based on pre-filtered quiz sessions for a single learner.
    """
    engagement = pd.Series(index=pd.Index(assessment_periods), dtype='object').fillna('')

    if learner_sessions.empty:
        engagement.update(pd.Series({period: "O" if len(quizzes_due_per_assessment_period.get(period, [])) > 0 else ""
                                      for period in assessment_periods}))
        return tuple(engagement.values)

    unique_block_ids_learner = set(learner_sessions['block_id'].unique())
    assessment_periods_engagement = {period: "" for period in assessment_periods}

    for period, elements_due in quizzes_due_per_assessment_period.items():
        if not elements_due:
            continue

        quiz_sessions_in_period = learner_sessions[learner_sessions['end_time'] <= period.right]
        unique_block_ids_period = set(quiz_sessions_in_period['block_id'].unique())

        quizzes_done_on_time = sum(1 for element in elements_due if element in unique_block_ids_period)
        quizzes_done = sum(1 for element in elements_due if element in unique_block_ids_learner)

        if course_has_due_dates:
            if quizzes_done_on_time == len(elements_due):
                assessment_periods_engagement[period] = "T"
            elif quizzes_done_on_time + quizzes_done == len(elements_due):
                assessment_periods_engagement[period] = "B"
            elif quizzes_done_on_time > 0 or quizzes_done > 0:
                assessment_periods_engagement[period] = "A"
            else:
                assessment_periods_engagement[period] = "O"
        else:
            if quizzes_done == len(elements_due):
                assessment_periods_engagement[period] = "T"
            elif quizzes_done > 0:
                assessment_periods_engagement[period] = "A"
            else:
                assessment_periods_engagement[period] = "O"

    for period in assessment_periods:
        engagement[period] = assessment_periods_engagement[period]
    return tuple(engagement.values)

def construct_learner_engagement_mapping(quiz_sessions_df: pd.DataFrame, metadata_df: pd.DataFrame, course_learner_df: pd.DataFrame, video_interactions_df: pd.DataFrame, submissions_df: pd.DataFrame, ora_sessions_df: pd.DataFrame, course_id: str, course_has_due_dates: bool) -> Tuple[Counter[tuple, int], Counter[tuple, int]]:    
    course_start_date = pd.Timestamp(metadata_df["object"]["start_date"])
   
    f = open("./mandatory_elements_per_course.json", "r")
    all_courses_mandatory_elements = json.load(f)
    f.close()

    all_mandatory_elements = all_courses_mandatory_elements[course_id]
    elements_per_week = {}
    order_map: dict = metadata_df["object"]["order_map"]
    child_parent_map = metadata_df["object"]["child_parent_map"]

    for element in all_mandatory_elements:
        if element is None:
            continue
        parent = child_parent_map.get(element)
        while parent is not None and "chapter" not in parent:
            parent = child_parent_map.get(parent)
        
        if parent is None:
            continue
        week_number = order_map.get(parent)
        if week_number not in elements_per_week:
            elements_per_week[week_number] = []
        elements_per_week[week_number].append(element)

    interval_dict = {}
    for week, elements in elements_per_week.items():
        start_date = course_start_date + pd.DateOffset(weeks=week - 1)
        end_date = start_date + pd.DateOffset(weeks=1)
        interval = pd.Interval(left=start_date, right=end_date, closed='left')
        interval_dict[interval] = elements
    elements_per_assessment_period = pd.Series(interval_dict)        

    quizzes_due_per_assessment_period = elements_per_assessment_period.apply(
        lambda elements: [element for element in elements if "vertical+block" in element] or None).dropna()
    submissions_due_per_assessment_period = elements_per_assessment_period.apply(
        lambda elements: [element for element in elements if "problem+block" in element] or None).dropna()
    ora_due_per_assessment_period = elements_per_assessment_period.apply(
        lambda elements: [element for element in elements if "openassessment" in element] or None).dropna()

    assessment_periods = elements_per_assessment_period.index

    mooc_and_run_id = course_id.split(":")[1]

    empty_tuple = ("",) * len(assessment_periods)
    course_learner_ddf: DaskDataFrame = dd.from_pandas(course_learner_df, npartitions=6)
                                       
    if not quiz_sessions_df.empty:
        quiz_sessions_grouped_by_id = quiz_sessions_df.groupby('course_learner_id')
        with ProgressBar():
            course_learner_ddf['quiz_engagement'] = course_learner_ddf['course_learner_id'].apply(
                lambda learner_id: calculate_quiz_engagement(quiz_sessions_grouped_by_id.get_group(learner_id) or pd.DataFrame(),
                                                            quizzes_due_per_assessment_period,
                                                            assessment_periods,
                                                            course_has_due_dates)
,
                meta=('quiz_engagement', 'object')
            )

            course_learner_ddf = course_learner_ddf.compute()
    else:
        course_learner_ddf['quiz_engagement'] = [empty_tuple for _ in range(len(course_learner_df))]

    if not video_interactions_df.empty:
        video_interactions_by_id = video_interactions_df.groupby('course_learner_id')
        with ProgressBar():
            course_learner_ddf['video_engagement'] = course_learner_ddf['course_learner_id'].apply(
                lambda learner_id: calculate_video_engagement(video_interactions_by_id.get_group(learner_id) or pd.DataFrame(), metadata_df, assessment_periods),
                meta=('video_engagement', 'object')
            )
            course_learner_ddf = course_learner_ddf.compute()
    else:
        course_learner_ddf['video_engagement'] = [empty_tuple for _ in range(len(course_learner_df))]


    if not submissions_df.empty:
        submissions_by_id = submissions_df.groupby('course_learner_id')
        with ProgressBar():
            course_learner_ddf['submission_engagement'] = course_learner_ddf['course_learner_id'].apply(
                lambda learner_id: calculate_submission_engagement(submissions_by_id.get_group(learner_id) or pd.DataFrame(), submissions_due_per_assessment_period, assessment_periods, course_has_due_dates)
                if learner_id in submissions_by_id.groups else (("",) * len(assessment_periods)),
                meta=('submission_engagement', 'object')
            )
            course_learner_ddf = course_learner_ddf.compute()
    else:
        course_learner_ddf['submission_engagement'] = [empty_tuple for _ in range(len(course_learner_df))]

    if not ora_sessions_df.empty:
        ora_sessions_by_id = ora_sessions_df.groupby('course_learner_id')
        with ProgressBar():
            course_learner_ddf['ora_engagement'] = course_learner_ddf['course_learner_id'].apply(
                lambda learner_id: calculate_ora_engagement(ora_sessions_by_id.get_group(learner_id) or pd.DataFrame(), ora_due_per_assessment_period, assessment_periods, course_has_due_dates),
                meta=('ora_engagement', 'object')
                )
    else:
        course_learner_ddf['ora_engagement'] = [empty_tuple for _ in range(len(course_learner_df))]


    # print(course_learner_df['quiz_engagement'].unique())
    # print(course_learner_df['video_engagement'].unique())
    # print(course_learner_df['submission_engagement'].unique())
    # print(course_learner_df['ora_engagement'].unique())
    # print(course_learner_df.head())

    # print all rows where quiz_engagement is not a tuple consisting of 7 strings
    # print(course_learner_df[~course_learner_df['quiz_engagement'].apply(lambda x: isinstance(x, tuple) and len(x) == 7 and all(isinstance(i, str) for i in x))])

    course_learner_ddf['engagement'] = course_learner_ddf.apply(
        lambda row: process_engagement(row['quiz_engagement'], row['video_engagement'], row['submission_engagement'], row['ora_engagement'], elements_per_assessment_period
                                       ), axis=1)
    # print(course_learner_df.head())

    # Dump course_learner_df to a file
    course_learner_ddf.to_csv(f"./{mooc_and_run_id}_learner_engagement.csv", index=False)

    male_counts = course_learner_df[course_learner_df['gender']
                                    == 'm']['engagement'].value_counts()
    # print(male_counts)
    female_counts = course_learner_df[course_learner_df['gender']
                                      == 'f']['engagement'].value_counts()

    return male_counts, female_counts


def process_engagement(quiz_engagement: tuple[str], video_engagement: tuple[str], submission_engagement: tuple[str], ora_engagement: tuple[str], assessment_periods: tuple[str]) -> tuple[str]:
    engagement = tuple()
    assert len(quiz_engagement) == len(video_engagement) == len(submission_engagement) == len(ora_engagement) == len(assessment_periods)

    for idx, assessment_periods in enumerate(assessment_periods):

        quiz_engagement_for_period = quiz_engagement[idx] or ""
        video_engagement_for_period = video_engagement[idx] or ""
        submission_engagement_for_period = submission_engagement[idx] or ""
        ora_engagement_for_period = ora_engagement[idx] or ""

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
        # Rule 4: If there are no non-video statuses (meaning there are no deadlines), they are on track if they have watched lectures from that week.
        elif not non_video_statuses and video_engagement_for_period == "A":
            engagement += ("T",)
        # Rule 4: If all non-video engagements are "O", check video engagement
        else:
            # If video_engagement is 'A', return 'A'. Otherwise, return 'O' since video_engagement can only be 'A' or 'O'.
            engagement += ("A" if video_engagement_for_period == "A" else "O",)

    return engagement


def get_learner_quiz_sessions_before_due_date(learner_quiz_sessions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    filtered_quiz_sessions = learner_quiz_sessions[learner_quiz_sessions['end_time'] < assessment_period.right]    
    return filtered_quiz_sessions


def get_learner_ora_sessions_per_assessment_period(learner_ora_sessions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    filtered_ora_sessions = learner_ora_sessions[learner_ora_sessions['end_time'] < assessment_period.right]
    return filtered_ora_sessions


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


def calculate_k_means_clusters(trajectory_frequency: pd.Series, course_run: str, gender: str) -> None:
    learner_engagement_lists = []
# Iterate over each pattern and frequency
    for pattern, freq in trajectory_frequency.items():
        # Convert pattern to numeric
        numeric_values = engagement_to_numeric(pattern)
        # Repeat the entire list of numeric values for the number of frequencies
        # This assumes each frequency is a different learner's data for simplicity; adjust logic as needed
        for _ in range(freq):
            learner_engagement_lists.append(numeric_values)
    X = np.array(learner_engagement_lists)

    best_score = -1
    best_kmeans = None

    for _ in tqdm(range(100), desc=f"Finding best KMeans clusters for {course_run}"):
        kmeans = KMeans(n_clusters=4, random_state=None).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_score = score
            best_kmeans = kmeans

    clusters = best_kmeans.labels_
    centroids = best_kmeans.cluster_centers_

    df = pd.DataFrame(X, columns=[f'Week {i+1}' for i in range(X.shape[1])])
    df['cluster'] = clusters

    print(f"\n{course_run}")
    print(f"Best silhouette score: {best_score:.4f}")

    print("\nCluster sizes:")
    print(df['cluster'].value_counts().sort_index())

    for cluster in sorted(df['cluster'].unique()):
        print(f"\nAverage engagement scores for Cluster {cluster}:")
        print(df[df['cluster'] == cluster].iloc[:, :-1].mean())

    plt.figure(figsize=(10, 6))
    for i, centroid in enumerate(centroids):
        plt.plot(centroid, label=f'Cluster {i}', marker='o', linestyle='--')
    plt.title(f'{gender} Cluster Centroids for {course_run}')
    plt.xlabel('Week')
    plt.ylabel('Engagement Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(X.shape[1]), [f'Week {i+1}' for i in range(X.shape[1])])
    plt.savefig(f'{gender} {course_run.split(":")[1]}.png')
    plt.close()


def get_quiz_sessions(db: Database, course_id: str, filtered_ids: set[str]) -> pd.DataFrame:
    quiz_sessions_query = construct_query("course_id", course_id, filtered_ids, exact=True)
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
    db = client["edx_test"]

    courses = get_courses(db)

    for _, course in courses.iterrows():
        try:
            full_course_id = course["course_id"]
            # if "ST1x" not in full_course_id or "EX101x+2T2018" in full_course_id or "EX101x+3T2015" in full_course_id or "EX101x+3T2016" in full_course_id:
            #     continue

            if "EX101x+2T2018" in full_course_id:
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
            course_has_due_dates = has_due_dates(metadata)
            
            course_learner = get_course_learner(
                db, escaped_course_id)

            course_learner = course_learner.merge(get_learner_demographic(
                db, full_course_id, set(course_learner['course_learner_id']))[['course_learner_id', 'gender']], on='course_learner_id', how='left')

            filtered_learners = course_learner[course_learner['gender'].isin([
                'm', 'f'])]

            filtered_ids: set[str] = set(filtered_learners['course_learner_id'].unique())

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

            print("All data loaded")
            male_learner_engagement_mapping, female_learner_engagement_mapping = construct_learner_engagement_mapping(quiz_sessions, metadata,
                                                                                                                      course_learner, video_interactions, submissions, ora_sessions, full_course_id, course_has_due_dates)
            
            course_run = full_course_id.split(":")[1].split("+")
            title = f"{course_run[1]}_{course_run[2]}"
            calculate_k_means_clusters(
                male_learner_engagement_mapping, title, "Male")
            calculate_k_means_clusters(
                female_learner_engagement_mapping, title, "Female")

        except Exception as ex:
            print(''.join(traceback.format_exception(type(ex),
                                                     value=ex, tb=ex.__traceback__)))


if __name__ == '__main__':
    main()
