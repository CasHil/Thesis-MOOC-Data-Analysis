import pandas as pd
import pymongo
import os
import json
from dotenv import load_dotenv


def categorize_seek_rate_change(events):
    categorized_events = []
    for i in range(len(events)):
        event = events[i]
        if event[1] == 'seek_video':
            if i > 0 and events[i-1][1] == 'seek_video' and event[0] - events[i-1][0] < 1:
                categorized_events[-1][1] = 'SSf' if event[2]['new_time'] > event[2]['old_time'] else 'SSb'
            else:
                event[1] = 'Sf' if event[2]['new_time'] > event[2]['old_time'] else 'Sb'
        elif event[1] == 'speed_change_video':
            event[1] = 'Rf' if event[2]['new_speed'] > event[2]['old_speed'] else 'Rs'
        categorized_events.append(event)
    return categorized_events

# Function to create clickstream sequences


def create_clickstream_sequences(data, event_mapping):
    clickstream_data = []

    for course_entry in data:
        course_id = course_entry['course_id']
        learner_id = course_entry['course_learner_id']
        videos = course_entry['videos']

        for video_id, events in videos.items():
            session_events = []
            for event in events:
                event_time, event_type, additional_info = event
                if event_type == 'load_video':
                    if session_events:
                        session_events = categorize_seek_rate_change(
                            session_events)
                        clickstream = ''.join(
                            [event[1] for event in session_events if event[1] in event_mapping.values()])
                        clickstream_data.append({
                            'course_id': course_id,
                            'learner_id': learner_id,
                            'video_id': video_id,
                            'clickstream': clickstream
                        })
                    session_events = []  # Start a new session
                elif event_type in event_mapping:
                    mapped_event_type = event_mapping[event_type]
                    session_events.append(
                        [event_time, mapped_event_type, additional_info])

            if session_events:
                session_events = categorize_seek_rate_change(session_events)
                clickstream = ''.join(
                    [event[1] for event in session_events if event[1] in event_mapping.values()])
                clickstream_data.append({
                    'course_id': course_id,
                    'learner_id': learner_id,
                    'video_id': video_id,
                    'clickstream': clickstream
                })

    return pd.DataFrame(clickstream_data)


def main():
    load_dotenv()
    courses = json.loads(os.getenv('COURSES'))
    client = pymongo.MongoClient('localhost', 27017)
    db = client['edx_prod']
    collection = db['video_engagement_sessions']

    event_mapping = {
        'pause_video': 'Pa',
        'play_video': 'Pl',
        'seek_video': 'Sf',  # initial mapping
        'speed_change_video': 'Rf',  # initial mapping
        'stop_video': 'stop_video',
    }

    for course in courses:
        video_engagement_sessions = collection.find(
            {'course_id': {"$regex": course}})
        video_engagement_sessions_df = pd.DataFrame(video_engagement_sessions)
        # Find unique course ids
        course_runs = set([entry['course_id']
                          for entry in video_engagement_sessions])
        for course_run in course_runs:
            course_data = video_engagement_sessions_df[video_engagement_sessions_df['course_id'] == course_run]
            clickstream_sequences_df = create_clickstream_sequences(
                course_data, event_mapping)
            print(clickstream_sequences_df)

        clickstream_sequences_df = create_clickstream_sequences(
            video_engagement_sessions, event_mapping)

    # Display the DataFrame
    print(clickstream_sequences_df)
