from pymongo import MongoClient
from tqdm import tqdm

def find_ora_id(session_id: str):
    parts = session_id.split('_')
    course_id = parts[2].split(':')[1]
    block_id = parts[3]
    new_id = f"block-v1:{course_id}+type@openassessment+block@{block_id}"
    return new_id


def transform_course_id(original_id: str):
    parts = original_id.split('_')
    new_id = f"course-v1:DelftX+{parts[0]}+{parts[1]}"
    return new_id


def main():
    client = MongoClient('mongodb://localhost:27017/')
    db = client["edx_prod"]

    quiz_sessions = db.quiz_sessions
    total_quiz_documents = quiz_sessions.count_documents({})

    for document in tqdm(quiz_sessions.find(), total=total_quiz_documents):
        session_id = document.get("sessionId", "")
        course_learner_id = document.get("course_learner_id", "")
        try:
            block_id = session_id.split("_")[2]
            course_id = course_learner_id.split("_")[0]
        except IndexError:
            block_id = ""
            course_id = ""

        if block_id and course_id:
            quiz_sessions.update_one(
                {"_id": document["_id"]},
                {"$set": {"block_id": block_id, "course_id": course_id}}
            )


    ora_sessions = db.ora_sessions
    total_ora_documents = ora_sessions.count_documents({})
        

    for document in tqdm(ora_sessions.find(), total=total_ora_documents):
        session_id = document.get("sessionId", "")
        course_learner_id = document.get("course_learner_id", "")
        try:
            block_id = find_ora_id(session_id)
            course_id = course_learner_id.split("_")[0]
        except IndexError:
            block_id = ""
            course_id = ""

        if block_id and course_id:
            ora_sessions.update_one(
                {"_id": document["_id"]},
                {"$set": {"block_id": block_id, "course_id": course_id}}
            )

    submissions = db.submissions
    total_submission_documents = submissions.count_documents({})
    for document in tqdm(submissions.find(), total=total_submission_documents):
        try:
            course_learner_id = document.get("course_learner_id", "")
            course_id = course_learner_id.split("_")[0]
        except IndexError:
            course_id = ""

        if course_id:
            submissions.update_one(
                {"_id": document["_id"]},
                {"$set": {"course_id": course_id}}
            )


    course_learner = db.course_learner
    total_course_learner_documents = course_learner.count_documents({})
    for document in tqdm(course_learner.find(), total=total_course_learner_documents):
        try:
            course_learner_id = document.get("course_learner_id", "")
            course_id = course_learner_id.split("_")[0]
        except IndexError:
            course_id = ""

        if course_id:
            course_learner.update_one(
                {"_id": document["_id"]},
                {"$set": {"course_id": course_id}}
            )

    video_interactions = db.video_interactions
    total_video_interactions_documents = video_interactions.count_documents({})
    for document in tqdm(video_interactions.find(), total=total_video_interactions_documents):
        try:
            course_learner_id = document.get("course_learner_id", "")
            course_id = course_learner_id.split("_")[0]
        except IndexError:
            course_id = ""

        if course_id:
            video_interactions.update_one(
                {"_id": document["_id"]},
                {"$set": {"course_id": course_id}}
            )



    metadata = db.metadata
    total_metadata_documents = metadata.count_documents({})
    for document in tqdm(metadata.find(), total=total_metadata_documents):
        try:
            course_id = document.get("name", "")
            course_id = transform_course_id(course_id)
            
        except IndexError:
            course_id = ""

        if course_id:
            metadata.update_one(
                {"_id": document["_id"]},
                {"$set": {"course_id": course_id}}
            )

    learner_demographic = db.learner_demographic
    total_learner_demographic_documents = learner_demographic.count_documents({})
    for document in tqdm(learner_demographic.find(), total=total_learner_demographic_documents):
        try:
            course_learner_id = document.get("course_learner_id", "")
            course_id = course_learner_id.split("_")[0]
        except IndexError:
            course_id = ""

        if course_id:
            learner_demographic.update_one(
                {"_id": document["_id"]},
                {"$set": {"course_id": course_id}}
            )

    video_engagement_sessions = db.video_engagement_sessions
    total_video_engagement_sessions_documents = video_engagement_sessions.count_documents({})
    for document in tqdm(video_engagement_sessions.find(), total=total_video_engagement_sessions_documents):
        try:
            course_learner_id = document.get("learner_id", "")
            course_id = course_learner_id.split("_")[0]
        except IndexError:
            course_id = ""

        if course_id:
            video_engagement_sessions.update_one(
                {"_id": document["_id"]},
                {
                    "$rename": {"learner_id": "course_learner_id"},
                    "$set": {"course_id": course_id}
                }
            )

            
    print("Update complete.")
