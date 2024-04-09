# from datetime import datetime
# import json
# import os
# import glob
# import sqlite3


# def extract_course_information(files):
#     course_metadata_map = {}
#     for file in files:
#         file_name = file["key"]
#         if "course_structure" not in file_name:
#             if file == files[-1]:
#                 print("Course structure file is missing!")
#                 return course_metadata_map
#             continue

#         child_parent_map = {}
#         element_time_map = {}
#         element_time_map_due = {}
#         element_type_map = {}
#         element_without_time = []
#         quiz_question_map = {}
#         block_type_map = {}
#         order_map = {}
#         element_name_map = {}

#         json_object = json.loads(file["value"])
#         for record in json_object:
#             if json_object[record]["category"] == "course":
#                 course_id = record.replace("block-", "course-").replace(
#                     "+type@course+block@course", "").replace("i4x://", "").replace("course/", "")
#                 course_metadata_map["course_id"] = course_id
#                 course_metadata_map["course_name"] = json_object[record]["metadata"]["display_name"]
#                 start_date = json_object[record]["metadata"]["start"]
#                 end_date = json_object[record]["metadata"]["end"]
#                 course_metadata_map["start_time"] = start_date
#                 course_metadata_map["end_time"] = end_date
#                 element_position = 0
#                 for child in json_object[record]["children"]:
#                     element_position += 1
#                     child_parent_map[child] = record
#                     order_map[child] = element_position
#                 element_time_map[record] = start_date
#                 element_type_map[record] = json_object[record]["category"]
#             else:
#                 element_id = record
#                 element_name_map[element_id] = json_object[element_id]["metadata"].get(
#                     "display_name", "")
#                 element_position = 0
#                 for child in json_object[element_id].get("children", []):
#                     element_position += 1
#                     child_parent_map[child] = element_id
#                     order_map[child] = element_position
#                 if "start" in json_object[element_id]["metadata"]:
#                     element_time_map[element_id] = json_object[element_id]["metadata"]["start"]
#                 else:
#                     element_without_time.append(element_id)
#                 if "due" in json_object[element_id]["metadata"]:
#                     element_time_map_due[element_id] = json_object[element_id]["metadata"]["due"]
#                 element_type_map[element_id] = json_object[element_id]["category"]
#                 if json_object[element_id]["category"] == "problem":
#                     quiz_question_map[element_id] = json_object[element_id]["metadata"].get(
#                         "weight", 1.0)
#                 if json_object[element_id]["category"] == "sequential":
#                     block_type_map[element_id] = json_object[element_id]["metadata"].get(
#                         "display_name", "")

#         for element_id in element_without_time:
#             element_start_time = element_time_map.get(
#                 child_parent_map[element_id], "")
#             element_time_map[element_id] = element_start_time

#         course_metadata_map["element_time_map"] = element_time_map
#         course_metadata_map["element_time_map_due"] = element_time_map_due
#         course_metadata_map["element_type_map"] = element_type_map
#         course_metadata_map["quiz_question_map"] = quiz_question_map
#         course_metadata_map["child_parent_map"] = child_parent_map
#         course_metadata_map["block_type_map"] = block_type_map
#         course_metadata_map["order_map"] = order_map
#         course_metadata_map["element_name_map"] = element_name_map
#         print("Metadata map ready")
#         return course_metadata_map


# def process_metadata_files(files, connection):
#     course_metadata_map = extract_course_information(files)
#     course_record = [{
#         "course_id": course_metadata_map["course_id"],
#         "course_name": course_metadata_map["course_name"],
#         "start_time": course_metadata_map["start_time"],
#         "end_time": course_metadata_map["end_time"],
#     }]
#     sql_insert("courses", course_record, connection)


# def read_metadata_files(directory_path, connection):
#     processed_files = []
#     for filename in os.listdir(directory_path):
#         if filename.endswith((".sql", ".json", ".mongo")):
#             filepath = os.path.join(directory_path, filename)
#             with open(filepath, mode='r', encoding='utf-8') as f:
#                 content = f.read()
#                 processed_files.append({"key": filename, "value": content})
#     process_metadata_files(processed_files, connection)


# def compare_datetime(a_datetime, b_datetime):
#     a_datetime = datetime.fromisoformat(a_datetime)
#     b_datetime = datetime.fromisoformat(b_datetime)
#     if a_datetime < b_datetime:
#         return -1
#     elif a_datetime > b_datetime:
#         return 1
#     else:
#         return 0


# def process_enrollment(course_id, input_file, course_metadata_map):
#     course_learner_map = {}
#     learner_enrollment_time_map = {}
#     enrolled_learner_set = set()
#     learner_index_record = []
#     learner_mode_map = {}

#     lines = input_file.split("\n")
#     for line in lines[1:]:
#         record = line.split("\t")
#         if len(record) < 2:
#             continue
#         active = record[4]
#         if active == "0":
#             continue
#         global_learner_id = record[0]
#         time = datetime.fromisoformat(record[2])
#         course_learner_id = course_id + "_" + global_learner_id
#         mode = record[4]
#         if compare_datetime(course_metadata_map["end_time"], time) == 1:
#             enrolled_learner_set.add(global_learner_id)
#             array = [global_learner_id, course_id, course_learner_id]
#             learner_index_record.append(array)
#             course_learner_map[global_learner_id] = course_learner_id
#             learner_enrollment_time_map[global_learner_id] = time
#             learner_mode_map[global_learner_id] = mode

#     return {
#         "course_learner_map": course_learner_map,
#         "learner_enrollment_time_map": learner_enrollment_time_map,
#         "enrolled_learner_set": enrolled_learner_set,
#         "learner_index_record": learner_index_record,
#         "learner_mode_map": learner_mode_map,
#     }


# def process_certificates(input_file, enrollment_values, course_metadata_map):
#     uncertified_learners = 0
#     certified_learners = 0

#     course_learner_record = []
#     certificate_map = {}

#     for line in input_file.split("\n"):
#         record = line.split("\t")
#         if len(record) < 7:
#             continue
#         global_learner_id, final_grade, certificate_status = record[0], record[1], record[3]
#         if global_learner_id in enrollment_values['course_learner_map']:
#             certificate_map[global_learner_id] = {
#                 'final_grade': final_grade,
#                 'certificate_status': certificate_status,
#             }

#             for global_learner_id, value in certificate_map.items():
#                 if value["certificate_status"] == "downloadable":
#                     course_learner_id = enrollment_values['course_learner_map'][global_learner_id]
#                     final_grade = value["final_grade"]
#                     enrollment_mode = enrollment_values['learner_mode_map'][global_learner_id]
#                     certificate_status = value["certificate_status"]
#                     register_time = enrollment_values['learner_enrollment_time_map'][global_learner_id]
#                     segment = enrollment_values['learner_segment_map'][global_learner_id]
#                     array = [
#                         course_learner_id,
#                         final_grade,
#                         enrollment_mode,
#                         certificate_status,
#                         register_time,
#                         segment,
#                     ]
#                     course_learner_record.append(array)
#                     certified_learners += 1
#                 else:
#                     uncertified_learners += 1
#         else:
#             for global_learner_id, course_learner_id in enrollment_values['course_learner_map'].items():
#                 final_grade = None
#                 enrollment_mode = enrollment_values['learner_mode_map'][global_learner_id]
#                 certificate_status = None
#                 register_time = enrollment_values['learner_enrollment_time_map'][global_learner_id]
#                 segment = enrollment_values['learner_segment_map'][global_learner_id]
#                 if global_learner_id in certificate_map:
#                     final_grade = certificate_map[global_learner_id]["final_grade"]
#                     certificate_status = certificate_map[global_learner_id]["certificate_status"]
#                 array = [
#                     course_learner_id,
#                     final_grade,
#                     enrollment_mode,
#                     certificate_status,
#                     register_time,
#                     segment,
#                 ]
#                 if certificate_status == "downloadable":
#                     certified_learners += 1
#                 else:
#                     uncertified_learners += 1
#                 course_learner_record.append(array)
#                 if register_time <= course_metadata_map['end_date']:
#                     if certificate_status == "downloadable":
#                         certified_learners += 1
#                     else:
#                         uncertified_learners += 1
#                 course_learner_record.append(array)

#     return {
#         'certifiedLearners': certified_learners,
#         'uncertifiedLearners': uncertified_learners,
#         'courseLearnerRecord': course_learner_record,
#     }


# def process_auth_map(input_file, enrollment_values):
#     learner_auth_map = {}
#     for line in input_file.split("\n"):
#         record = line.split("\t")
#         if record[0] in enrollment_values['enrolled_learner_set']:
#             learner_auth_map[record[0]] = {
#                 'mail': record[4],
#                 'staff': record[6]
#             }
#     return learner_auth_map


# def process_groups(course_id, input_file, enrollment_values):
#     group_map = {}
#     for line in input_file.split("\n"):
#         record = line.split("\t")
#         if len(record) < 4:
#             continue
#         global_learner_id = record[0]
#         group_type = record[2]
#         group_name = record[3]
#         course_learner_id = f"{course_id}_{global_learner_id}"
#         if global_learner_id in enrollment_values['enrolled_learner_set']:
#             group_map[course_learner_id] = [group_type, group_name]
#     return group_map


# def process_demographics(course_id, input_file, enrollment_values, learner_auth_map):
#     learner_demographic_record = []
#     for line in input_file.split("\n"):
#         record = line.split("\t")
#         if len(record) < 7:
#             continue
#         global_learner_id, gender, year_of_birth, level_of_education, country = record[
#             0], record[2], record[3], record[4], record[6]
#         course_learner_id = f"{course_id}_{global_learner_id}"
#         if global_learner_id in enrollment_values['enrolled_learner_set']:
#             learner_mail = learner_auth_map.get(
#                 global_learner_id, {}).get('mail', '')
#             array = [
#                 course_learner_id,
#                 gender,
#                 year_of_birth,
#                 level_of_education,
#                 country,
#                 learner_mail,
#                 enrollment_values['learner_segment_map'].get(
#                     global_learner_id, ''),
#             ]
#             learner_demographic_record.append(array)
#     return {'learnerDemographicRecord': learner_demographic_record}


# def process_forum_posting_interaction(forum_file, course_metadata_map):
#     forum_interaction_records = []
#     lines = forum_file.split("\n")
#     for line in lines:
#         if len(line) < 9:  # Assuming this is to check for a minimum line length
#             continue
#         jsonObject = json.loads(line)
#         post_id = jsonObject["_id"]["$oid"]
#         course_learner_id = jsonObject["course_id"] + \
#             "_" + jsonObject["author_id"]

#         post_type = jsonObject["_type"]
#         if post_type == "CommentThread":
#             post_type += "_" + jsonObject["thread_type"]
#         if "parent_id" in jsonObject and jsonObject["parent_id"] != "":
#             post_type = "Comment_Reply"

#         post_title = jsonObject.get("title", "")
#         if post_title:  # Add quotes if title exists
#             post_title = f'"{post_title}"'

#         post_content = f'"{jsonObject["body"]}"'
#         post_timestamp = datetime.strptime(
#             jsonObject["created_at"], "%Y-%m-%dT%H:%M:%SZ")

#         post_parent_id = jsonObject.get("parent_id", {}).get("$oid", "")
#         post_thread_id = jsonObject.get(
#             "comment_thread_id", {}).get("$oid", "")
#         array = [
#             post_id,
#             course_learner_id,
#             post_type,
#             post_title,
#             # Assuming escapeString function is for escaping newlines and such
#             post_content.replace('\n', '\\n').replace('\r', ''),
#             post_timestamp,
#             post_parent_id,
#             post_thread_id,
#         ]
#         if post_timestamp < datetime.strptime(course_metadata_map["end_time"], "%Y-%m-%dT%H:%M:%SZ"):
#             forum_interaction_records.append(array)
#     return forum_interaction_records
