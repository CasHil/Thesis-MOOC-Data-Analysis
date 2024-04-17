from pymongo import MongoClient
from tqdm import tqdm

client = MongoClient('mongodb://localhost:27017/')
db = client["edx_testing"]
collection = db.quiz_sessions
total_documents = collection.count_documents({})
for document in tqdm(collection.find(), total=total_documents):
    session_id = document.get("sessionId", "")
    course_learner_id = document.get("course_learner_id", "")
    try:
        block_id = session_id.split("_")[2]
        course_id = course_learner_id.split("_")[0]
    except IndexError:
        block_id = ""
        course_id = ""

    if block_id and course_id:
        collection.update_one(
            {"_id": document["_id"]},
            {"$set": {"block_id": block_id, "course_id": course_id}}
        )

print("Update complete.")
