from datetime import datetime
from datetime import datetime, timedelta
import sqlite3
import json
import gzip
import os
import asyncio
from typing import Any, Callable
import traceback

from dotenv import load_dotenv

load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
MOOC_DB_LOCATION = os.getenv('MOOC_DB_LOCATION')
COURSES = json.loads(os.getenv('COURSES'))


def process_null(input_string):
    if isinstance(input_string, str):
        if input_string == "" or input_string.upper() == "NULL":
            return None
        else:
            return input_string
    else:
        return input_string


def get_next_day(current_day):
    if isinstance(current_day, datetime):
        return current_day + timedelta(days=1)
    else:
        raise TypeError("current_day must be a datetime object")


def coucourse_elements_finder_string(eventlog_item, course_id):
    elements_id = ""
    course_id_filtered = course_id.split(":")[1] if len(
        course_id.split(":")) > 1 else course_id

    if "+type@" in eventlog_item and "block-v1:" in eventlog_item:
        templist = eventlog_item.split("/")
        for tempstring in templist:
            if "+type@" in tempstring and "block-v1:" in tempstring:
                elements_id = tempstring

    if elements_id == "" and "courseware/" in eventlog_item:
        templist = eventlog_item.split("/")
        tempflag = False
        for tempstring in templist:
            if tempstring == "courseware":
                tempflag = True
            elif tempflag and tempstring != "":
                elements_id = f"""block-v1:{
                    course_id_filtered}+type@chapter+block@{tempstring}"""
                break

    return elements_id


def course_elements_finder(eventlog, course_id):
    elements_id = coucourse_elements_finder_string(
        eventlog.get("event_type", ""), course_id)
    if elements_id == "":
        elements_id = coucourse_elements_finder_string(
            eventlog.get("path", ""), course_id)
    if elements_id == "":
        elements_id = coucourse_elements_finder_string(
            eventlog.get("page", ""), course_id)
    if elements_id == "":
        elements_id = coucourse_elements_finder_string(
            eventlog.get("referer", ""), course_id)
    return elements_id


def get_ora_event_type_and_element(full_event):
    event_type = ""
    element = ""
    meta = False
    if "openassessmentblock" in full_event.get("event_type", ""):
        event_type = full_event["event_type"].split(".")[-1]
        element = full_event["context"]["module"]["usage_key"].split("@")[-1]
    elif "openassessment+block" in full_event.get("event_type", ""):
        event_type = full_event["event_type"].split("/")[-1]
        element = full_event["event_type"].split("@")[-1].split("/")[0]
        meta = True

    ora_info = {
        "eventType": event_type,
        "element": element,
        "meta": meta,
    }
    return ora_info


def sql_log_insert(table: str, rows_array: list[dict], connection: sqlite3.Connection):
    if not rows_array:
        return
    placeholders = ', '.join(['?' for _ in rows_array[0]])
    query = f"INSERT INTO {table} VALUES ({placeholders})"
    with open("insert.sql", "w") as f:
        f.write(query)

    data_to_insert = []
    for row in rows_array:
        data_row = []
        row = list(row)
        for value in row:
            if isinstance(value, datetime):
                value = value.isoformat()
            data_row.append(value)
        data_to_insert.append(tuple(data_row))

    cursor = connection.cursor()
    cursor.executemany(query, data_to_insert)
    connection.commit()

    if cursor.rowcount > 0:
        today = datetime.now()
        time = today.strftime("%H:%M:%S.%f")
        print(f"Successfully added: {table} at {time}")


def process_general_sessions(
    course_metadata_map: dict[str, Any],
    chunk: str,
    file_index: int,
    total_files: int,
    chunk_index: int,
    connection: sqlite3.Connection
) -> None:
    current_course_id = course_metadata_map["course_id"]
    current_course_id = current_course_id[current_course_id.find(
        "+") + 1: current_course_id.rfind("+") + 7]

    learner_all_event_logs = {}
    updated_learner_all_event_logs = {}
    session_record = []
    learner_all_event_logs = dict(updated_learner_all_event_logs)
    updated_learner_all_event_logs = {}

    course_learner_id_set = set()
    lines = chunk.split("\n")
    # print("Starting general session processing")
    # print("  for file", log_file)
    # print("    with ", len(lines), "lines")
    for line in lines:
        if len(line) < 10 or current_course_id not in line:
            continue
        jsonObject = json.loads(line)

        if jsonObject is None or "user_id" not in jsonObject["context"]:
            continue
        global_learner_id = jsonObject["context"]["user_id"]
        event_type = jsonObject["event_type"]

        if global_learner_id != "":
            course_id = jsonObject["context"]["course_id"]
            course_learner_id = f"{course_id}_{global_learner_id}"
            event_time = string_to_datetime(jsonObject["time"])

            if course_learner_id in course_learner_id_set:
                learner_all_event_logs[course_learner_id].append(
                    {"event_time": event_time, "event_type": event_type})
            else:
                learner_all_event_logs[course_learner_id] = [
                    {"event_time": event_time, "event_type": event_type}]
                course_learner_id_set.add(course_learner_id)

    # print(learner_all_event_logs)
    for course_learner_id, event_logs in learner_all_event_logs.items():
        event_logs.sort(key=lambda x: x["event_time"])
        session_id, start_time, end_time, final_time = None, None, None, None
        for log in event_logs:
            event_time = string_to_datetime(log["event_time"])
            if start_time is None:
                start_time = event_time
                end_time = event_time
            else:
                verification_time = end_time + timedelta(minutes=30)
                if log["event_time"] > verification_time:
                    session_id = f"{course_learner_id}_{
                        start_time.timestamp()}_{end_time.timestamp()}"
                    duration = (end_time - start_time).total_seconds()
                    if duration > 5:
                        session_record.append(
                            [session_id, course_learner_id, start_time, end_time, duration])
                    final_time = event_time
                    session_id = ""
                    start_time, end_time = event_time, event_time
                else:
                    if log["event_type"] == "page_close":
                        end_time = event_time
                        session_id = f"{course_learner_id}_{
                            start_time.timestamp()}_{end_time.timestamp()}"
                        duration = (
                            end_time - start_time).total_seconds()
                        if duration > 5:
                            session_record.append(
                                [session_id, course_learner_id, start_time, end_time, duration])
                        session_id = ""
                        start_time, end_time = None, None
                        final_time = event_time
                    else:
                        end_time = event_time
        if final_time is not None:
            updated_learner_all_event_logs[course_learner_id] = [
                log for log in event_logs if event_time >= final_time]

    updated_session_record = []
    session_id_set = set()
    for array in session_record:
        session_id = array[0]
        if session_id not in session_id_set:
            session_id_set.add(session_id)
            updated_session_record.append(array)
    session_record = updated_session_record

    if session_record:
        data = []
        for session in session_record:
            session_id, course_learner_id, start_time, end_time, duration = session
            if chunk_index != 0:
                session_id += f"_{chunk_index}"
            if file_index != 0:
                session_id += f"_{file_index}"
            values = (session_id, course_learner_id,
                      start_time, end_time, duration)
            data.append(values)
        sql_log_insert("sessions", data, connection)
    else:
        print("no session info", file_index, total_files)


def process_video_interaction_sessions(
    course_metadata_map: dict[str, any],
    chunk: str,
    index: int,
    total: int,
    chunk_index: int,
    connection: sqlite3.Connection
) -> None:
    course_id = course_metadata_map["course_id"]
    current_course_id = course_id[course_id.find(
        "+") + 1: course_id.rfind("+") + 7]

    video_event_types = [
        "hide_transcript", "edx.video.transcript.hidden", "edx.video.closed_captions.hidden",
        "edx.video.closed_captions.shown", "load_video", "edx.video.loaded", "pause_video", "edx.video.paused",
        "play_video", "edx.video.played", "seek_video", "edx.video.position.changed", "show_transcript",
        "edx.video.transcript.shown", "speed_change_video", "stop_video", "edx.video.stopped", "video_hide_cc_menu",
        "edx.video.language_menu.hidden", "video_show_cc_menu", "edx.video.language_menu.shown"
    ]
    video_interaction_records = []
    video_interaction_map = {}
    learner_video_event_logs = {}
    updated_learner_video_event_logs = {}
    learner_video_event_logs = updated_learner_video_event_logs
    course_learner_id_set = set()

    if len(learner_video_event_logs) > 0:
        for course_learner_id in learner_video_event_logs:
            course_learner_id_set.add(course_learner_id)

    lines = chunk.split("\n")

    for line in lines:
        if len(line) < 10 or current_course_id not in line:
            continue
        jsonObject = json.loads(line)
        if jsonObject["event_type"] in video_event_types:
            if "user_id" not in jsonObject["context"]:
                continue
            global_learner_id = jsonObject["context"]["user_id"]
            if global_learner_id != "":
                course_id = jsonObject["context"]["course_id"]
                course_learner_id = f"{course_id}_{global_learner_id}"
                event_time = string_to_datetime(jsonObject["time"])
                event_type = jsonObject["event_type"]

                event_jsonObject = jsonObject["event"]
                if isinstance(event_jsonObject, str):
                    event_jsonObject = json.loads(event_jsonObject)

                video_id = event_jsonObject["id"].replace(
                    "-", "://").replace("-", "/")
                new_time = string_to_datetime(
                    event_jsonObject.get("new_time", 0))
                old_time = string_to_datetime(
                    event_jsonObject.get("old_time", 0))
                new_speed = event_jsonObject.get("new_speed", 0)
                old_speed = event_jsonObject.get("old_speed", 0)

                if event_type in ["seek_video", "edx.video.position.changed"] and new_time is not None and old_time is not None:
                    if course_learner_id in learner_video_event_logs:
                        learner_video_event_logs[course_learner_id].append({
                            "event_time": event_time,
                            "event_type": event_type,
                            "video_id": video_id,
                            "new_time": new_time,
                            "old_time": old_time,
                        })
                    else:
                        learner_video_event_logs[course_learner_id] = [{
                            "event_time": event_time,
                            "event_type": event_type,
                            "video_id": video_id,
                            "new_time": new_time,
                            "old_time": old_time,
                        }]
                        course_learner_id_set.add(course_learner_id)
                    continue

                if event_type == "speed_change_video":
                    if course_learner_id in learner_video_event_logs:
                        learner_video_event_logs[course_learner_id].append({
                            "event_time": event_time,
                            "event_type": event_type,
                            "video_id": video_id,
                            "new_speed": new_speed,
                            "old_speed": old_speed,
                        })
                    else:
                        learner_video_event_logs[course_learner_id] = [{
                            "event_time": event_time,
                            "event_type": event_type,
                            "video_id": video_id,
                            "new_speed": new_speed,
                            "old_speed": old_speed,
                        }]
                        course_learner_id_set.add(course_learner_id)
                    continue

                if course_learner_id in learner_video_event_logs:
                    learner_video_event_logs[course_learner_id].append({
                        "event_time": event_time,
                        "event_type": event_type,
                        "video_id": video_id,
                    })
                else:
                    learner_video_event_logs[course_learner_id] = [{
                        "event_time": event_time,
                        "event_type": event_type,
                        "video_id": video_id,
                    }]
                    course_learner_id_set.add(course_learner_id)
        if jsonObject["event_type"] not in video_event_types:
            if "user_id" not in jsonObject["context"]:
                continue
            global_learner_id = jsonObject["context"]["user_id"]
            if global_learner_id != "":
                course_id = jsonObject["context"]["course_id"]
                course_learner_id = f"{course_id}_{global_learner_id}"
                event_time = string_to_datetime(jsonObject["time"])
                event_type = jsonObject["event_type"]

                if course_learner_id in learner_video_event_logs:
                    learner_video_event_logs[course_learner_id].append({
                        "event_time": event_time,
                        "event_type": event_type
                    })
                else:
                    learner_video_event_logs[course_learner_id] = [{
                        "event_time": event_time,
                        "event_type": event_type
                    }]
                    course_learner_id_set.add(course_learner_id)
            for course_learner_id, event_logs in learner_video_event_logs.items():
                for log in event_logs:
                    log["event_time"] = string_to_datetime(log["event_time"])
            for course_learner_id, event_logs in learner_video_event_logs.items():
                event_logs.sort(key=lambda x: x["event_time"])
                video_id = ""
                video_start_time = None
                final_time = None
                times_forward_seek = 0
                duration_forward_seek = 0
                times_backward_seek = 0
                duration_backward_seek = 0
                speed_change_last_time = ""
                times_speed_up = 0
                times_speed_down = 0
                pause_check = False
                pause_start_time = None
                duration_pause = 0

                for log in event_logs:
                    if log["event_type"] in ["play_video", "edx.video.played"]:
                        video_start_time = log["event_time"]
                        video_id = log["video_id"]
                        if pause_check:
                            duration_pause = (
                                log["event_time"] - pause_start_time).total_seconds()
                            video_interaction_id = f"{course_learner_id}_{
                                video_id}_{int(pause_start_time.timestamp())}"
                            if duration_pause > 2 and duration_pause < 600:
                                if video_interaction_id in video_interaction_map:
                                    video_interaction_map[video_interaction_id]["times_pause"] = 1
                                    video_interaction_map[video_interaction_id]["duration_pause"] = duration_pause
                            pause_check = False
                        continue

                    if video_start_time is not None:
                        event_time = string_to_datetime(log["event_time"])
                        verification_time = video_start_time + \
                            timedelta(minutes=30)
                        if event_time > verification_time:
                            video_start_time = None
                            video_id = ""
                            final_time = event_time
                        else:
                            if log["event_type"] in ["seek_video", "edx.video.position.changed"] and video_id == log["video_id"]:
                                old_time = string_to_datetime(log["old_time"])
                                new_time = string_to_datetime(log["new_time"])

                                if new_time > old_time:
                                    times_forward_seek += 1
                                    duration_forward_seek += new_time - \
                                        old_time
                                if new_time < old_time:
                                    times_backward_seek += 1
                                    duration_backward_seek += old_time - \
                                        new_time
                                continue

                            if log["event_type"] == "speed_change_video" and video_id == log["video_id"]:
                                if speed_change_last_time == "":
                                    speed_change_last_time = log["event_time"]
                                    old_speed = log["old_speed"]
                                    new_speed = log["new_speed"]
                                    if old_speed < new_speed:
                                        times_speed_up += 1
                                    if old_speed > new_speed:
                                        times_speed_down += 1
                                else:
                                    if (log["event_time"] - speed_change_last_time).total_seconds() > 10:
                                        old_speed = log["old_speed"]
                                        new_speed = log["new_speed"]
                                        if old_speed < new_speed:
                                            times_speed_up += 1
                                        if old_speed > new_speed:
                                            times_speed_down += 1
                                    speed_change_last_time = log["event_time"]
                                continue
                            if log["event_type"] in ["pause_video", "edx.video.paused", "stop_video", "edx.video.stopped"] and video_id == log["video_id"]:
                                watch_duration = (
                                    log["event_time"] - video_start_time).total_seconds()
                                video_end_time = log["event_time"]
                                video_interaction_id = f"{course_learner_id}_{
                                    video_id}_{int(video_end_time.timestamp())}"
                                if watch_duration > 5:
                                    video_interaction_map[video_interaction_id] = {
                                        "course_learner_id": course_learner_id,
                                        "video_id": video_id,
                                        "type": "video",
                                        "watch_duration": watch_duration,
                                        "times_forward_seek": times_forward_seek,
                                        "duration_forward_seek": duration_forward_seek,
                                        "times_backward_seek": times_backward_seek,
                                        "duration_backward_seek": duration_backward_seek,
                                        "times_speed_up": times_speed_up,
                                        "times_speed_down": times_speed_down,
                                        "start_time": video_start_time,
                                        "end_time": video_end_time,
                                    }
                                if log["event_type"] in ["pause_video", "edx.video.paused"]:
                                    pause_check = True
                                    pause_start_time = video_end_time
                                times_forward_seek = 0
                                duration_forward_seek = 0
                                times_backward_seek = 0
                                duration_backward_seek = 0
                                speed_change_last_time = ""
                                times_speed_up = 0
                                times_speed_down = 0
                                video_start_time = None
                                video_id = ""
                                final_time = log["event_time"]

                            if not any(evt_type in log["event_type"] for evt_type in video_event_types):
                                video_end_time = log["event_time"]
                                watch_duration = (
                                    video_end_time - video_start_time).total_seconds()
                                video_interaction_id = f"{course_learner_id}_{
                                    video_id}_{int(video_end_time.timestamp())}_{chunk}"
                                if watch_duration > 5:
                                    video_interaction_map[video_interaction_id] = {
                                        "course_learner_id": course_learner_id,
                                        "video_id": video_id,
                                        "type": "video",
                                        "watch_duration": watch_duration,
                                        "times_forward_seek": times_forward_seek,
                                        "duration_forward_seek": duration_forward_seek,
                                        "times_backward_seek": times_backward_seek,
                                        "duration_backward_seek": duration_backward_seek,
                                        "times_speed_up": times_speed_up,
                                        "times_speed_down": times_speed_down,
                                        "start_time": video_start_time,
                                        "end_time": video_end_time,
                                    }
                                times_forward_seek = 0
                                duration_forward_seek = 0
                                times_backward_seek = 0
                                duration_backward_seek = 0
                                speed_change_last_time = ""
                                times_speed_up = 0
                                times_speed_down = 0
                                video_start_time = None
                                video_id = ""
                                final_time = log["event_time"]
                if final_time is not None:
                    new_logs = [
                        log for log in event_logs if log["event_time"] > final_time]
                    updated_learner_video_event_logs[course_learner_id] = new_logs

    video_interaction_record = []
    for interaction_id, interaction_info in video_interaction_map.items():
        times_pause = interaction_info.get("times_pause", 0)
        duration_pause = interaction_info.get("duration_pause", 0)
        array = [
            interaction_id,
            interaction_info["course_learner_id"],
            interaction_info["video_id"],
            round(interaction_info["watch_duration"]),
            round(interaction_info.get("times_forward_seek", 0)),
            round(interaction_info.get("duration_forward_seek", 0)),
            round(interaction_info.get("times_backward_seek", 0)),
            round(interaction_info.get("duration_backward_seek", 0)),
            round(interaction_info.get("times_speed_up", 0)),
            round(interaction_info.get("times_speed_down", 0)),
            round(times_pause),
            round(duration_pause),
            interaction_info["start_time"],
            interaction_info["end_time"],
        ]
        video_interaction_record.append(array)

    if video_interaction_record:
        data = []
        for array in video_interaction_record:
            interaction_id = f"{array[0]}_{index}_{
                chunk}" if chunk and index else array[0]
            values = {
                "interaction_id": interaction_id,
                "course_learner_id": array[1],
                "video_id": array[2],
                "duration": array[3],
                "times_forward_seek": array[4],
                "duration_forward_seek": array[5],
                "times_backward_seek": array[6],
                "duration_backward_seek": array[7],
                "times_speed_up": array[8],
                "times_speed_down": array[9],
                "times_pause": array[10],
                "duration_pause": array[11],
                "start_time": array[12],
                "end_time": array[13],
            }
            data.append(values)

    if connection is not None:
        sql_log_insert("video_interactions",
                       video_interaction_records, connection)
    else:
        print("no video interaction info", index, total)


def string_to_datetime(input_string: str) -> datetime:
    if not isinstance(input_string, str):
        return input_string
    try:
        return datetime.strptime(input_string, "%Y-%m-%dT%H:%M:%S.%f%z")
    except ValueError:
        return datetime.strptime(input_string, "%Y-%m-%dT%H:%M:%S%z")


def process_assessments_submissions(
    course_metadata_map: dict[str, Any],
    chunk: str,
    index: int,
    total: int,
    chunk_index: int,
    connection: sqlite3.Connection
) -> None:
    course_id = course_metadata_map["course_id"]
    current_course_id = course_id[course_id.find(
        "+") + 1: course_id.rfind("+") + 7]

    # print("Starting quiz processing")
    submission_event_collection = ["problem_check"]
    submission_uni_index = chunk_index * 100

    submission_data = []
    assessment_data = []

    lines = chunk.split("\n")
    for line in lines:
        if len(line) < 10 or current_course_id not in line:
            continue
        jsonObject = json.loads(line)
        if jsonObject["event_type"] in submission_event_collection and "user_id" in jsonObject["context"]:
            global_learner_id = jsonObject["context"]["user_id"]
            if global_learner_id:
                course_learner_id = f"""{
                    jsonObject['context']['course_id']}_{global_learner_id}"""
                event_time = string_to_datetime(jsonObject["time"])

                if isinstance(jsonObject["event"], dict):
                    event_data = jsonObject["event"]
                    question_id = event_data.get("problem_id", "")
                    grade = event_data.get("grade", "")
                    max_grade = event_data.get("max_grade", "")

                if question_id:
                    submission_id = f"{course_learner_id}_{
                        question_id}_{submission_uni_index}"
                    submission_uni_index += 1

                    submission_record = {
                        "submission_id": submission_id,
                        "course_learner_id": course_learner_id,
                        "question_id": question_id,
                        "submission_timestamp": event_time,
                    }
                    submission_data.append(submission_record)

                    if grade and max_grade:
                        assessment_record = {
                            "assessment_id": submission_id,
                            "course_learner_id": course_learner_id,
                            "max_grade": max_grade,
                            "grade": grade,
                        }
                        assessment_data.append(assessment_record)

    if assessment_data:
        sql_log_insert("assessments", assessment_data, connection)
        print(f"Processed {len(assessment_data)} assessment records")
    else:
        print(f"No assessment data for file {index} of {total}")

    if submission_data:
        sql_log_insert("submissions", submission_data, connection)
        print(f"Processed {len(submission_data)} submission records")
    else:
        print(f"No submission data for file {index} of {total}")


def process_ora_sessions(course_metadata_map: dict[str, Any], chunk: str, index: int, total: int, total_chunks: int, connection: sqlite3.Connection, prep_log_file_fn: Callable[[int, list[str], dict[str, Any], sqlite3.Connection], None], files: list[str]):
    course_id = course_metadata_map["course_id"]
    current_course_id = course_id[course_id.find(
        "+") + 1: course_id.rfind("+") + 7]

    # print("Starting ORA sessions")
    course_metadata_map = {}
    child_parent_map = course_metadata_map["child_parent_map"]
    learner_all_event_logs = {}
    updated_learner_all_event_logs = {}
    ora_sessions = {}
    ora_events = {}
    ora_sessions_record = []

    learner_all_event_logs = updated_learner_all_event_logs.copy()
    updated_learner_all_event_logs = {}
    # course_learner_id_set = set(learner_all_event_logs.keys())

    lines = chunk.split("\n")
    for line in lines:
        if len(line) < 10 or current_course_id not in line:
            continue
        jsonObject = json.loads(line)
        if "user_id" not in jsonObject["context"]:
            continue
        global_learner_id = jsonObject["context"]["user_id"]
        if global_learner_id == "":
            continue

        course_id = jsonObject["context"]["course_id"]
        course_learner_id = f"{course_id}_{global_learner_id}"
        event_time = string_to_datetime(jsonObject["time"])
        event = {
            "event_time": event_time,
            "event_type": jsonObject["event_type"],
            "full_event": jsonObject,
        }
        learner_all_event_logs.setdefault(
            course_learner_id, []).append(event)

    for course_learner_id, event_logs in learner_all_event_logs.items():
        event_logs.sort(key=lambda x: x["event_time"])
        sessionId = ""
        startTime = None
        endTime = None
        finalTime = None
        currentStatus = ""
        currentElement = ""
        saveCount = 0
        peerAssessmentCount = 0
        selfAssessed = False
        submitted = False
        eventType = ""
        meta = False

        learner_ora_events = []
        for event_log in event_logs:
            if "openassessment" in event_log["event_type"]:
                if not sessionId:
                    startTime = event_log["event_time"]
                    endTime = event_log["event_time"]
                    eventDetails = get_ora_event_type_and_element(
                        event_log["full_event"])
                    currentElement = eventDetails["element"]
                    eventType = eventDetails["eventType"]
                    meta = eventDetails["meta"]
                    sessionId = f"ora_session_{
                        course_learner_id}_{currentElement}"

                    if meta and not currentStatus:
                        currentStatus = "viewed"
                    elif eventType == "save_submission":
                        saveCount += 1
                        currentStatus = "saved"
                    elif eventType == "create_submission":
                        submitted = True
                        currentStatus = "submitted"
                    elif eventType == "self_assess":
                        selfAssessed = True
                        currentStatus = "selfAssessed"
                    elif eventType == "peer_assess" and not meta:
                        peerAssessmentCount += 1
                        currentStatus = "assessingPeers"

                    learner_ora_events.append(
                        f"Empty id: {currentStatus}_{meta}_{eventType}")
            else:
                if "openassessment" in event_log["event_type"]:
                    previous_element = current_element
                    event_details = get_ora_event_type_and_element(
                        event_log["full_event"])
                    current_element = event_details["element"]
                    event_type = event_details["event_type"]
                    meta = event_details["meta"]

                    verification_time = endTime + timedelta(minutes=30)
                    if event_log["event_time"] > verification_time:
                        session_id = f"{session_id}_{int(startTime.timestamp())}_{
                            int(endTime.timestamp())}"

                        duration = (
                            endTime - startTime).total_seconds()
                        if duration > 5:
                            ora_sessions_record.append({
                                "session_id": session_id,
                                "course_learner_id": course_learner_id,
                                "save_count": save_count,
                                "peer_assessment_count": peer_assessment_count,
                                "submitted": submitted,
                                "self_assessed": self_assessed,
                                "start_time": startTime,
                                "end_time": endTime,
                                "duration": duration,
                                "current_element": current_element
                            })

                        final_time = event_log["event_time"]
                        learner_ora_events.append(f"Over 30 min, to store: {
                            current_status}_{meta}_{event_type}")

                        session_id = f"ora_session_{
                            course_learner_id}_{current_element}"
                        startTime = event_log["event_time"]
                        endTime = event_log["event_time"]
                        if meta and current_status == "":
                            current_status = "viewed"
                        elif event_type == "save_submission":
                            save_count += 1
                            current_status = "saved"
                        elif event_type == "create_submission":
                            submitted = True
                            current_status = "submitted"
                        elif event_type == "self_assess":
                            self_assessed = True
                            current_status = "selfAssessed"
                        elif event_type == "peer_assess" and not meta:
                            peer_assessment_count += 1
                            current_status = "assessingPeers"

                        learner_ora_events.append(
                            f"Over 30 min, new: {current_status}_{meta}_{event_type}")
                    else:
                        endTime = event_log["event_time"]
                        if meta and current_status == "":
                            current_status = "viewed"
                        elif event_type == "save_submission":
                            save_count += 1
                            current_status = "saved"
                        elif event_type == "create_submission":
                            submitted = True
                            current_status = "submitted"
                        elif event_type == "self_assess":
                            self_assessed = True
                            current_status = "selfAssessed"
                        elif event_type == "peer_assess" and not meta:
                            peer_assessment_count += 1
                            current_status = "assessingPeers"

                        learner_ora_events.append(
                            f"Under 30 min: {current_status}_{meta}_{event_type}")
                else:
                    learner_ora_events.append(f"""Not ORA, to store: {
                        current_status}_{meta}_{event_type}""")

                    if event_log["event_time"] <= endTime + timedelta(minutes=30):
                        endTime = event_log["event_time"]
                    session_id = f"{session_id}_{int(startTime.timestamp())}_{
                        int(endTime.timestamp())}"

                    duration = (endTime - startTime).total_seconds()
                    if duration > 5:
                        ora_sessions_record.append({
                            "session_id": session_id,
                            "course_learner_id": course_learner_id,
                            "save_count": save_count,
                            "peer_assessment_count": peer_assessment_count,
                            "submitted": submitted,
                            "self_assessed": self_assessed,
                            "start_time": startTime,
                            "end_time": endTime,
                            "duration": duration,
                            "current_element": current_element
                        })

                    final_time = event_log["event_time"]
                    session_id = ""
                    startTime = None
                    endTime = None
                    meta = False
                    event_type = ""
                    save_count = 0
                    self_assessed = False
                    peer_assessment_count = 0

    if learner_ora_events:
        ora_events[course_learner_id] = learner_ora_events
    if not ora_sessions_record:
        print(f"no ORA session info for file {index} of {total}")
        return

    data = []
    for record in ora_sessions_record:
        session_id = record["session_id"]
        if chunk != 0:
            session_id += f"_{chunk}"
        if index != 0:
            session_id += f"_{index}"

        values = {
            "session_id": session_id,
            "course_learner_id": record["course_learner_id"],
            "times_save": record["save_count"],
            "times_peer_assess": record["peer_assessment_count"],
            "submitted": record["submitted"],
            "self_assessed": record["self_assessed"],
            "start_time": record["start_time"].isoformat(),
            "end_time": record["end_time"].isoformat(),
            "duration": record["duration"],
            "assessment_id": record["current_element"],
        }
        data.append(values)

    sql_log_insert("ora_sessions", data, connection)
    print(f"""Sending {len(data)} ORA sessions to storage at {
        datetime.now()}""")
    if chunk == total_chunks:
        index += 1
        prep_log_file_fn(index, files, course_metadata_map, connection)


def process_quiz_sessions(course_metadata_map: dict[str, Any], log_file: str, index: int, total: int, chunk: int, connection: sqlite3.Connection):
    current_course_id = course_metadata_map["course_id"].split(
        "+")[1] + "+" + course_metadata_map["course_id"].split("+")[2][:6]

    submission_event_collection = [
        "problem_check", "save_problem_check", "problem_check_fail", "save_problem_check_fail",
        "problem_graded", "problem_rescore", "problem_rescore_fail", "problem_reset", "reset_problem",
        "reset_problem_fail", "problem_save", "save_problem_fail", "save_problem_success", "problem_show",
        "showanswer"
    ]

    child_parent_map = course_metadata_map["child_parent_map"]
    learner_all_event_logs = {}
    updated_learner_all_event_logs = {}
    quiz_sessions = {}

    # print("Starting quiz sessions")

    learner_all_event_logs = updated_learner_all_event_logs.copy()
    updated_learner_all_event_logs.clear()
    course_learner_id_set = set()

    lines = chunk.split("\n")
    for line in lines:
        if len(line) < 10 or current_course_id not in line:
            continue

        jsonObject = json.loads(line)
        if "user_id" not in jsonObject["context"]:
            continue

        global_learner_id = jsonObject["context"]["user_id"]
        event_type = jsonObject["event_type"]
        if global_learner_id == "":
            continue

        course_id = jsonObject["context"]["course_id"]
        course_learner_id = f"{course_id}_{global_learner_id}"
        event_time = string_to_datetime(jsonObject["time"])

        if course_learner_id in learner_all_event_logs:
            learner_all_event_logs[course_learner_id].append(
                {"event_time": event_time, "event_type": event_type})
        else:
            learner_all_event_logs[course_learner_id] = [
                {"event_time": event_time, "event_type": event_type}]

        for course_learner_id, event_logs in learner_all_event_logs.items():
            event_logs.sort(key=lambda x: x['event_type'])
            event_logs.sort(key=lambda x: x['event_time'])

            session_id = ""
            start_time = None
            end_time = None
            final_time = None
            for event in event_logs:
                if not session_id:
                    if any(sub in event["event_type"] for sub in ["problem+block", "_problem;_"]) or \
                            event["event_type"] in submission_event_collection:
                        question_id = ""
                        if "problem+block" in event["event_type"]:
                            question_id = event["event_type"].split(
                                "/")[4]
                        elif "_problem;_" in event["event_type"]:
                            question_id = event["event_type"].split(
                                "/")[6].replace(";_", "/")

                        if question_id in child_parent_map:
                            session_id = f"quiz_session_{
                                child_parent_map[question_id]}_{course_learner_id}"
                            start_time = event["event_time"]
                            end_time = event["event_time"]
                else:
                    if any(sub in event["event_type"] for sub in ["problem+block", "_problem;_"]) or \
                            event["event_type"] in submission_event_collection:
                        verification_time = end_time + \
                            timedelta(minutes=30)
                        if event["event_time"] > verification_time:
                            if session_id in quiz_sessions:
                                quiz_sessions[session_id]["time_array"].append(
                                    {"start_time": start_time, "end_time": end_time})
                            else:
                                quiz_sessions[session_id] = {"course_learner_id": course_learner_id,
                                                             "time_array": [{"start_time": start_time, "end_time": end_time}]}

                            final_time = event["event_time"]
                            session_id, start_time, end_time = "", None, None
                        else:
                            end_time = event["event_time"]
                    else:
                        verification_time = end_time + \
                            timedelta(minutes=30)
                        if event["event_time"] <= verification_time:
                            end_time = event["event_time"]

                        if session_id in quiz_sessions:
                            quiz_sessions[session_id]["time_array"].append(
                                {"start_time": start_time, "end_time": end_time})
                        else:
                            quiz_sessions[session_id] = {"course_learner_id": course_learner_id,
                                                         "time_array": [{"start_time": start_time, "end_time": end_time}]}

                        final_time = event["event_time"]
                        session_id, start_time, end_time = "", None, None

                if final_time:
                    updated_learner_all_event_logs[course_learner_id] = [
                        log for log in event_logs if log["event_time"] >= final_time]

    for session_id, session_info in quiz_sessions.items():
        if len(session_info["time_array"]) > 1:
            updated_time_array = []
            start_time = None
            end_time = None

            for i, time_range in enumerate(session_info["time_array"]):
                if i == 0:
                    start_time = time_range["start_time"]
                    end_time = time_range["end_time"]
                else:
                    verification_time = end_time + timedelta(minutes=30)
                updated_time_array = []

                for i, time_range in enumerate(session_info["time_array"]):
                    if time_range["start_time"] > verification_time:
                        updated_time_array.append(
                            {"start_time": start_time, "end_time": end_time})
                        start_time = time_range["start_time"]
                        end_time = time_range["end_time"]
                    else:
                        end_time = time_range["end_time"]

                    if i == len(session_info["time_array"]) - 1:
                        updated_time_array.append(
                            {"start_time": start_time, "end_time": end_time})
                        quiz_sessions[session_id]["time_array"] = updated_time_array

    quiz_session_records = []
    for session_id, session_info in quiz_sessions.items():
        course_learner_id = session_info["course_learner_id"]
        for time_range in session_info["time_array"]:
            start_time = time_range["start_time"]
            end_time = time_range["end_time"]
            if start_time < end_time:
                duration = (end_time - start_time).total_seconds()
                final_session_id = f"{session_id}_{int(start_time.timestamp())}_{
                    int(end_time.timestamp())}"
                if duration > 5:
                    record = {
                        "session_id": final_session_id,
                        "course_learner_id": course_learner_id,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                    }
                    quiz_session_records.append(record)
    if not quiz_session_records:
        print("No quiz session data")
        return
    data = []
    for record in quiz_session_records:
        session_id = record["session_id"]
        if chunk != 0:
            session_id += f"_{chunk}"
        if index != 0:
            session_id += f"_{index}"

        record["session_id"] = session_id
        record["duration"] = process_null(record["duration"])
        data.append(record)

    if connection is None:
        return data

    # print("Quiz session data", data)
    sql_log_insert("quiz_sessions", data, connection)


async def unzip_and_chunk_logfile(file_name: str, file_index: int, total_files: int, process_unzipped_chunk_fn: Callable[[str, int, int, int, int, sqlite3.Connection, dict], None], course_metadata_map: dict, connection: sqlite3.Connection, files: list[str]):
    if not file_name.endswith(".gz"):
        print(f"{file_name} is not a log file (should end with: .log.gz)")
        return
    try:
        with gzip.open(file_name, 'rt', encoding='utf-8') as f:
            content = f.read()
            chunk_size = 30 * 1024 * 1024
            chunks = [content[i:i + chunk_size]
                      for i in range(0, len(content), chunk_size)]
            total_chunks = len(chunks)
            chunk_index = 0

            for chunk in chunks:
                print(f"Processing chunk {chunk_index + 1} of {total_chunks}")
                await process_unzipped_chunk_fn(chunk, file_index, total_files, chunk_index, total_chunks, connection, course_metadata_map, files)
                chunk_index += 1

    except Exception as e:
        traceback.print_exception(e)


async def process_unzipped_chunk(
    chunk: str,
    file_index: int,
    total_files: int,
    chunk_index: int,
    total_chunks: int,
    db_connection: sqlite3.Connection,
    course_metadata_map: dict[str, Any],
    files: list[str]
) -> None:
    print(f"""Processing file {
          file_index + 1}/{total_files} at {datetime.now().isoformat()}""")
    process_general_sessions(course_metadata_map, chunk,
                             file_index, total_files, chunk_index, db_connection)
    process_video_interaction_sessions(
        course_metadata_map, chunk, file_index, total_files, chunk_index, db_connection)
    process_assessments_submissions(
        course_metadata_map, chunk, file_index, total_files, chunk_index, db_connection)
    process_quiz_sessions(course_metadata_map, chunk,
                          file_index, total_files, chunk_index, db_connection)
    process_ora_sessions(course_metadata_map, chunk, file_index, total_files,
                         chunk_index, total_chunks, db_connection, prepare_log_files, files)


def identify_course(course_id):
    if "EX101x" in course_id:
        return "EX101x"
    elif "FP101x" in course_id:
        return "FP101x"
    elif "ST1x" in course_id:
        return "ST1x"
    elif "UnixTx" in course_id:
        return "UnixTx"
    else:
        raise ValueError("Course not found")


def identify_log_files_per_course_run(course_name: str) -> dict[str, list[str]]:
    file_ending = ".log.gz"
    log_files_per_course_run = {}

    for dir_name in os.listdir(WORKING_DIRECTORY):
        if course_name in dir_name:
            course_path = os.path.join(
                os.path.abspath(WORKING_DIRECTORY), dir_name)
            files = []
            for file_name in os.listdir(course_path):
                if file_name.endswith(file_ending):
                    files.append(os.path.join(course_path, file_name))

            if len(files) == 0:
                raise ValueError(f"No log files found for {course_name}")

            log_files_per_course_run[dir_name] = files

    return log_files_per_course_run


async def prepare_log_files(file_index: int, files: list[str], course_metadata_map: dict, connection: sqlite3.Connection):
    total_files = len(files)

    # if (file_index < total_files):
    # print("File", file_index + 1, "out of", total_files)
    for counter, f in enumerate(files):
        # print("File", counter + 1, "out of", total_files)
        if counter == file_index:
            today = datetime.now()
            # print("Starting at", today)
            await unzip_and_chunk_logfile(f, file_index,
                                          total_files, process_unzipped_chunk, course_metadata_map, connection, files)


async def main():
    connection = sqlite3.connect(MOOC_DB_LOCATION)
    # Clear the ora_sessions, sessions, assessments, quiz_sessions, video_interactions
    # tables before inserting new data

    for table in ["ora_sessions", "sessions", "assessments", "quiz_sessions", "video_interactions"]:
        cursor = connection.cursor()
        cursor.execute(f"DELETE FROM {table}")
        connection.commit()

    for course in COURSES:
        log_files = identify_log_files_per_course_run(course)
        for file_index, (course_run, files) in enumerate(log_files.items()):
            # print(f"""Processing {course_run} course, file {
            #       file_index + 1} of {len(log_files)}""")
            cursor = connection.cursor()
            cursor.execute(
                "SELECT object FROM metadata WHERE course_id=?", (course_run,))
            course_metadata_map = json.loads(cursor.fetchone()[0])
            await prepare_log_files(0, files, course_metadata_map, connection)


if __name__ == "__main__":
    asyncio.run(main())
