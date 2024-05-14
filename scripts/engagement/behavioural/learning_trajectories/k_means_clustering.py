import re
from typing import Tuple
import traceback
from collections import Counter
from functools import partial

import pandas as pd
import dask.dataframe as dd
from dask.dataframe import DataFrame as DaskDataFrame
import dask.array as da
from pandas.core.groupby import DataFrameGroupBy
from ast import literal_eval
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
    
    submissions_df = submissions_df.dropna(subset=['course_learner_id'])

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
    
    video_interactions_df = video_interactions_df.dropna(subset=['course_learner_id'])
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
    
    ora_sessions_df = ora_sessions_df.dropna(subset=['course_learner_id'])

    return downcast_numeric_columns(ora_sessions_df)


def process_element_id(x: str) -> str:
    return "+type@".join(x.split("+type@")[1:])


def get_learner_submissions_in_assessment_period(learner_submissions: pd.DataFrame, assessment_period: pd.Interval) -> pd.Series:
    learner_submissions_before_due_date = learner_submissions[learner_submissions['submission_timestamp'] < assessment_period.right]
    return learner_submissions_before_due_date



def calculate_submission_engagement(learner_submissions: pd.DataFrame, due_dates_per_assessment_period: pd.Series, assessment_periods: list[pd.Interval], course_has_due_dates: bool) -> Tuple[str]:

    engagement = pd.Series(index=pd.Index(assessment_periods), dtype='object').fillna('')
    if learner_submissions.empty:
        engagement.update(pd.Series({period: "O" if len(due_dates_per_assessment_period.get(period, [])) > 0 else "" for period in assessment_periods}))
        return tuple(engagement.values)
    
    if course_has_due_dates:
        elements_completed_on_time = set()
        elements_completed_late = set()

        for assessment_period in assessment_periods:
            elements_due_in_period = set(due_dates_per_assessment_period.get(assessment_period, []))
            period_submissions = learner_submissions[
                learner_submissions['question_id'].isin(elements_due_in_period) &
                (pd.to_datetime(learner_submissions['submission_timestamp']) <= assessment_period.right)
            ]

            elements_completed_on_time.update(period_submissions['question_id'].unique())

            late_submissions = learner_submissions[
                learner_submissions['question_id'].isin(elements_due_in_period) &
                (pd.to_datetime(learner_submissions['submission_timestamp']) > assessment_period.right)
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

    engagement = engagement.sort_index()
    return tuple(engagement.values)


def find_period(timestamp: pd.Timestamp, periods: list[pd.Interval]) -> pd.Interval:
    for period in periods:
        if timestamp in period:
            return period
    if timestamp > periods[-1].right:
        return periods[-1]
    elif timestamp < periods[0].left:
        return periods[0]

def calculate_ora_engagement(learner_ora_sessions: pd.DataFrame, ora_due_dates_per_assessment_period: pd.Series, assessment_periods: list[pd.Interval], course_has_due_dates: bool) -> Tuple[str]:
    engagement = pd.Series(index=pd.Index(assessment_periods), dtype='object').fillna('')

    if learner_ora_sessions.empty:
        engagement.update(pd.Series({period: "O" if len(ora_due_dates_per_assessment_period.get(period, [])) > 0 else "" for period in assessment_periods}))
        return tuple(engagement.values)

    if course_has_due_dates:
        assessment_periods_with_all_oras_on_time = set()
        assessment_periods_with_all_oras_late_or_on_time = set()
        assessment_periods_with_some_oras_done = set()
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
        assessment_periods_with_all_oras_done = set()
        assessment_periods_with_some_oras_done = set()

        for assessment_period, elements_due in ora_due_dates_per_assessment_period.items():
            if (learner_ora_sessions['block_id'].isin(elements_due)).all():
                assessment_periods_with_all_oras_done.add(assessment_period)
            elif (learner_ora_sessions['block_id'].isin(elements_due)).any():
                assessment_periods_with_some_oras_done.add(assessment_period)

        for period in assessment_periods:
            if period in assessment_periods_with_all_oras_done:
                engagement[period] = "T"
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

def calculate_video_engagement(learner_video_interactions: pd.DataFrame, metadata_df: pd.DataFrame, videos_per_assessment_period: pd.Series, assessment_periods: list[pd.Interval]) -> Tuple[str]:
    """
    Calculate the engagement based on pre-filtered video interactions for a single learner.
    """
    engagement = pd.Series(index=pd.Index(assessment_periods), dtype='object').fillna('')

    if learner_video_interactions.empty:
        engagement.update(pd.Series({period: "" for period in assessment_periods}))
        return tuple(engagement.values)

    # Create a copy of the DataFrame
    learner_video_interactions_copy = learner_video_interactions.copy()

    # Modify the copied DataFrame
    learner_video_interactions_copy.loc[:, 'chapter_start_time'] = learner_video_interactions_copy['video_id'].apply(
        lambda vid: get_chapter_start_time_by_video_id(metadata_df, vid)
    )   
    
    videos_grouped_by_period: DataFrameGroupBy = learner_video_interactions_copy.groupby(
        lambda idx: find_period(learner_video_interactions_copy.loc[idx, 'chapter_start_time'], assessment_periods)
    )

    for period in assessment_periods:
        videos_in_period = videos_per_assessment_period.get(period, [])
        if not videos_in_period:  # No videos in this assessment period
            engagement.loc[period] = ""
        elif period in videos_grouped_by_period.groups:  # Videos present in this assessment period
            engagement.loc[period] = "A"
        else:  # Videos present but learner didn't watch any
            engagement.loc[period] = "O"

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

def extract_week_number_from_element(element_id: str, child_parent_map: dict, order_map: dict) -> int:
    if not element_id or not child_parent_map or not order_map:
        return -1
    
    parent = child_parent_map.get(element_id)
    while "chapter" not in parent:
        parent = child_parent_map.get(parent)

    if not parent:
        return -1
    
    return order_map.get(parent)
    

def construct_learner_engagement_mapping(quiz_sessions_df: pd.DataFrame, metadata_df: pd.DataFrame, course_learner_df: pd.DataFrame, video_interactions_df: pd.DataFrame, submissions_df: pd.DataFrame, ora_sessions_df: pd.DataFrame, course_id: str, course_has_due_dates: bool) -> Tuple[Counter[tuple, int], Counter[tuple, int]]:    
    course_start_date = pd.Timestamp(metadata_df["object"]["start_date"])
   
    f = open("./mandatory_elements_per_course.json", "r")
    all_courses_mandatory_elements = json.load(f)
    f.close()

    all_mandatory_elements = all_courses_mandatory_elements[course_id]
    elements_per_week = {}
    order_map: dict = metadata_df["object"]["order_map"]
    child_parent_map: dict = metadata_df["object"]["child_parent_map"]

    video_keys = [key for key in order_map.keys() if "video" in key]
    openassessment_keys = [key for key in order_map.keys() if "openassessment" in key]

    elements = all_mandatory_elements + video_keys + openassessment_keys

    for element in tqdm(elements, desc=f"Finding the week that elements belong to"):
        week_number = extract_week_number_from_element(element, child_parent_map, order_map)
        if week_number == -1:
            continue
        if week_number not in elements_per_week:
            elements_per_week[week_number] = []
        elements_per_week[week_number].append(element)

    interval_dict = {}
    for week, elements in tqdm(elements_per_week.items(), desc=f"Grouping elements by assessment period"):
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
    videos_per_assessment_period = elements_per_assessment_period.apply(
        lambda elements: [element for element in elements if "video" in element] or None).dropna()

    assessment_periods = elements_per_assessment_period.index

    mooc_and_run_id = course_id.split(":")[1]

    # empty_tuple = ("",) * len(assessment_periods)
    # empty_tuple_series_pd = pd.Series([empty_tuple for _ in range(len(course_learner_df))])

    # if not submissions_df.empty:
    #     submissions_by_id = submissions_df.groupby('course_learner_id')
    #     submissions_engagement = course_learner_df.apply(
    #         lambda row: calculate_submission_engagement(
    #             submissions_by_id.get_group(row['course_learner_id']) if row['course_learner_id'] in submissions_by_id.groups else pd.DataFrame(),
    #             metadata_df,
    #             submissions_due_per_assessment_period,
    #             assessment_periods
    #         ),
    #         axis=1,
    #     )

    # else:
    #     submissions_engagement = empty_tuple_series_pd

    # course_learner_df['submission_engagement'] = submissions_engagement

    course_learner_ddf: DaskDataFrame = dd.from_pandas(course_learner_df, npartitions=6)
    empty_tuple = ("",) * len(assessment_periods)
    empty_tuple_series_pd = pd.Series([empty_tuple for _ in range(int(course_learner_ddf.shape[0].compute()))])
    empty_tuple_series_dd = dd.from_pandas(empty_tuple_series_pd, npartitions=course_learner_ddf.npartitions)

    print("Setting quiz engagement computations for Dask")
    if not quiz_sessions_df.empty:
        quiz_sessions_by_id = quiz_sessions_df.groupby('course_learner_id')
        quiz_engagement = course_learner_ddf.apply(
            lambda row: calculate_quiz_engagement(
                quiz_sessions_by_id.get_group(row['course_learner_id']) if row['course_learner_id'] in quiz_sessions_by_id.groups else pd.DataFrame(),
                quizzes_due_per_assessment_period,
                assessment_periods,
                course_has_due_dates
            ),
            axis=1,
            meta=('quiz_engagement', 'object')
        )
    else:
        quiz_engagement = empty_tuple_series_dd

    quiz_engagement = quiz_engagement.astype(object)
    course_learner_ddf = course_learner_ddf.assign(quiz_engagement=quiz_engagement)

    print("Setting video engagement computations for Dask")
    if not video_interactions_df.empty:
        video_interactions_by_id = video_interactions_df.groupby('course_learner_id')
        video_engagement = course_learner_ddf.apply(
            lambda row: calculate_video_engagement(
                video_interactions_by_id.get_group(row['course_learner_id']) if row['course_learner_id'] in video_interactions_by_id.groups else pd.DataFrame(),
                metadata_df,
                videos_per_assessment_period,
                assessment_periods
            ),
            axis=1,
            meta=('video_engagement', 'object')
        )
    else:
        video_engagement = empty_tuple_series_dd
    
    video_engagement = video_engagement.astype(object)
    course_learner_ddf['video_engagement'] = video_engagement

    print("Setting submission engagement computations for Dask")
    if not submissions_df.empty:
        submissions_by_id = submissions_df.groupby('course_learner_id')
        submission_engagement = course_learner_ddf.apply(
                lambda row: calculate_submission_engagement(
                    submissions_by_id.get_group(row['course_learner_id']) if row['course_learner_id'] in submissions_by_id.groups else pd.DataFrame(),
                    submissions_due_per_assessment_period,
                    assessment_periods,
                    course_has_due_dates
                ),
            axis=1,
            meta=('submission_engagement', 'object')
        )
    else:
        submission_engagement = empty_tuple_series_dd

    submission_engagement = submission_engagement.astype(object)
    course_learner_ddf = course_learner_ddf.assign(submission_engagement=submission_engagement)

    print("Setting ORA engagement computations for Dask")
    if not ora_sessions_df.empty:
        ora_sessions_by_id = ora_sessions_df.groupby('course_learner_id')
        ora_engagement = course_learner_ddf.apply(
                lambda row: calculate_ora_engagement(
                    ora_sessions_by_id.get_group(row['course_learner_id']) if row['course_learner_id'] in ora_sessions_by_id.groups else pd.DataFrame(),
                    ora_due_per_assessment_period,
                    assessment_periods,
                    course_has_due_dates
                ),
            axis=1,
            meta=('ora_engagement', 'object')
        )
    else:
        ora_engagement = empty_tuple_series_dd

    ora_engagement = ora_engagement.astype(object)
    course_learner_ddf = course_learner_ddf.assign(ora_engagement=ora_engagement)

    print("Running computations for video, quiz, submission, and ORA engagement")
    with ProgressBar():
        course_learner_df = course_learner_ddf.compute()
    
    course_learner_ddf = dd.from_pandas(course_learner_df, npartitions=6)

    engagement = course_learner_ddf.apply(
        lambda row: calculate_engagement(row['quiz_engagement'], row['video_engagement'], row['submission_engagement'], row['ora_engagement'], assessment_periods),
        axis=1,
        meta=('engagement', 'object')
    )

    course_learner_ddf = course_learner_ddf.assign(engagement=engagement)

    print("Running computations for engagement")
    with ProgressBar():
        course_learner_ddf = course_learner_ddf.compute()

    course_learner_ddf = course_learner_ddf.drop_duplicates(subset=['course_learner_id'])

    course_learner_ddf.to_csv(f"output/{mooc_and_run_id}_learner_engagement.csv", index=False)

    male_counts = course_learner_ddf[course_learner_ddf['gender']
                                    == 'm']['engagement'].value_counts()
    female_counts = course_learner_ddf[course_learner_ddf['gender']
                                      == 'f']['engagement'].value_counts()

    return male_counts, female_counts


def get_engagement_for_period(engagement: tuple[str], idx: int) -> str:
    try:
        engagement_for_period = engagement[idx]
        if not engagement_for_period in ("T", "B", "A", "O"):
            return ""
        return engagement_for_period
    except:
        return ""

def calculate_engagement(quiz_engagement: tuple[str], video_engagement: tuple[str], submission_engagement: tuple[str], ora_engagement: tuple[str], assessment_periods: pd.Series) -> tuple[str]:
    engagement = tuple()
    if isinstance(video_engagement, str):
        video_engagement = literal_eval(video_engagement)
    if isinstance(quiz_engagement, str):
        quiz_engagement = literal_eval(quiz_engagement)
    if isinstance(submission_engagement, str):
        submission_engagement = literal_eval(submission_engagement)
    if isinstance(ora_engagement, str):
        ora_engagement = literal_eval(ora_engagement)
    
    for idx, _ in enumerate(assessment_periods):
        quiz_engagement_for_period = get_engagement_for_period(quiz_engagement, idx)
        submission_engagement_for_period = get_engagement_for_period(submission_engagement, idx)
        ora_engagement_for_period = get_engagement_for_period(ora_engagement, idx)

        non_video_statuses = set([quiz_engagement_for_period,
                                  submission_engagement_for_period, ora_engagement_for_period])

        non_video_statuses.discard("")            
        video_engagement_for_period = get_engagement_for_period(video_engagement, idx)
        all_statuses = non_video_statuses.union({video_engagement_for_period})

        # Rule 1: If non_video_statuses include 'B' and may include 'T', but not 'A' or 'O'
        if non_video_statuses == {'B'} or non_video_statuses == {'B', 'T'}:
            engagement += ("B",)
        # Rule 2: If non_video_statuses are all 'T', return 'T'
        elif non_video_statuses == {'T'}:
            engagement += ("T",)
        # Rule 3: If non_video_statuses include a combination of "B", "O", "A", and "T", but not just "O", return 'A'
        elif set("BOAT").issuperset(all_statuses) and not all_statuses == {'O'}:
            engagement += ("A",)
        # Rule 4: If video_engagement_for_period is 'A', return 'T' if non_video_statuses is empty, otherwise 'A'
        elif video_engagement_for_period == "A":
            engagement += ("T",) if non_video_statuses == set() else ("A",)
        # Rule 5: If there are no due dates and no videos due, and non_video_statuses is empty, return 'T'
        elif not non_video_statuses and video_engagement_for_period == "":
            engagement += ("T",)
        # Rule 6: Default to 'O'
        else:
            engagement += ("O",)

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
    for pattern, freq in trajectory_frequency.items():
        numeric_values = engagement_to_numeric(pattern)
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

    # Print how many learners are in each cluster
    print("\nCluster sizes:")
    print(df['cluster'].value_counts().sort_index())

    # Save the average engagement socres for a gender and course run to a file. Also save the number of learners per cluster.
    with open(f"output/{course_run}_{gender}_clusters.txt", "w") as f:
        f.write(f"Best silhouette score: {best_score:.4f}\n\n")
        f.write("Cluster sizes:\n")
        f.write(f"{df['cluster'].value_counts().sort_index()}\n\n")
        for cluster in sorted(df['cluster'].unique()):
            f.write(f"Average engagement scores for Cluster {cluster}:\n")
            f.write(f"{df[df['cluster'] == cluster].iloc[:, :-1].mean()}\n\n")

    plt.figure(figsize=(10, 6))
    for i, centroid in enumerate(centroids):
        plt.plot(centroid, label=f'Cluster {i}', marker='o', linestyle='--')
    plt.title(f'{gender} Cluster Centroids for {course_run}')
    plt.xlabel('Week')
    plt.ylabel('Engagement Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(X.shape[1]), [f'Week {i+1}' for i in range(X.shape[1])])
    plt.savefig(f'figures/{gender}_{course_run}.png')
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
    
    quiz_sessions_df = quiz_sessions_df.dropna(subset=['course_learner_id'])

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

            if "EX101x+2T2018" in full_course_id or "EX101x+3T_2017" in full_course_id or "EX101x+3T2016" in full_course_id:
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

            # Only use 20 learners
            # filtered_learners = filtered_learners.head(20)

            filtered_ids: set[str] = set(filtered_learners['course_learner_id'].unique())

            course_id = "course-v1:DelftX+" + "+".join(course_id_with_course_run.split("_")[:2])
            submissions = get_submissions(db, course_id, filtered_ids)
            print("Submissions done")
            video_interactions = get_video_interactions(
                db, course_id, filtered_ids)
            print("Video interactions done")
            quiz_sessions = get_quiz_sessions(
                db, course_id, filtered_ids)
            print("Quiz sessions done")
            ora_sessions = get_ora_sessions(
                db, course_id, filtered_ids)

            print("ORA sessions done")
            male_learner_engagement_mapping, female_learner_engagement_mapping = construct_learner_engagement_mapping(quiz_sessions, metadata,
                                                                                                                      filtered_learners, video_interactions, submissions, ora_sessions, full_course_id, course_has_due_dates)
            
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
