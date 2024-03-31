CREATE TABLE certificates (
    hash_id VARCHAR(255) NOT NULL,
    grade NUMERIC(5,2),
    course_id VARCHAR(255) NOT NULL,
    status VARCHAR(50),
    created_date DATETIME NOT NULL,
    modified_date DATETIME NOT NULL,
    mode VARCHAR(50),
    PRIMARY KEY (hash_id, course_id),
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE course_instructors (
    hash_id VARCHAR(255) NOT NULL,
    org VARCHAR(255) NOT NULL,
    course_id VARCHAR(255) NOT NULL,
    role VARCHAR(100) NOT NULL,
    PRIMARY KEY (hash_id, course_id, role)
);

CREATE TABLE enrollments (
    hash_id VARCHAR(255) NOT NULL,
    course_id VARCHAR(255) NOT NULL,
    created DATETIME NOT NULL,
    is_active BOOLEAN NOT NULL,
    mode VARCHAR(50) NOT NULL,
    PRIMARY KEY (hash_id, course_id),
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE grades (
    hash_id VARCHAR(255) NOT NULL,
    course_id VARCHAR(255) NOT NULL,
    percent_grade NUMERIC(5,2),
    letter_grade VARCHAR(100),
    passed_timestamp DATETIME,
    created DATETIME,
    modified DATETIME,
    PRIMARY KEY (hash_id, course_id),
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE user_profiles (
    hash_id VARCHAR(255) NOT NULL,
    language VARCHAR(50),
    gender VARCHAR(10),
    year_of_birth INTEGER,
    level_of_education VARCHAR(50),
    goals TEXT,
    country VARCHAR(50),
    PRIMARY KEY (hash_id)
);

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    hash_id VARCHAR(255) NOT NULL,
    start_time TEXT,
    end_time TEXT,
    duration NUMERIC,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE forum_sessions (
    session_id TEXT PRIMARY KEY,
    hash_id VARCHAR(255) NOT NULL,
    times_search INTEGER,
    start_time TEXT,
    end_time TEXT,
    duration NUMERIC,
    relevent_element_id TEXT,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE video_interactions (
    interaction_id TEXT PRIMARY KEY,
    hash_id  NOT NULL,
    video_id TEXT NOT NULL,
    duration NUMERIC,
    times_forward_seek INTEGER,
    duration_forward_seek NUMERIC,
    times_backward_seek INTEGER,
    duration_backward_seek NUMERIC,
    times_speed_up INTEGER,
    times_speed_down INTEGER,
    times_pause INTEGER,
    duration_pause NUMERIC,
    start_time TEXT,
    end_time TEXT,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE assessments (
    assessment_id TEXT PRIMARY KEY,
    hash_id VARCHAR(255),
    max_grade NUMERIC,
    grade NUMERIC,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE submissions (
    submission_id TEXT PRIMARY KEY,
    hash_id VARCHAR(255) NOT NULL,
    question_id TEXT NOT NULL,
    submission_timestamp TEXT,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id),
    FOREIGN KEY (question_id) REFERENCES quiz_questions(question_id)
);

CREATE TABLE quiz_sessions (
    session_id TEXT PRIMARY KEY,
    hash_id VARCHAR(255) NOT NULL,
    start_time TEXT,
    end_time TEXT,
    duration NUMERIC,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE ora_sessions (
    session_id TEXT PRIMARY KEY,
    hash_id VARCHAR(255) NOT NULL,
    times_save INTEGER,
    times_peer_assess INTEGER,
    submitted BOOLEAN,
    self_assessed BOOLEAN,
    start_time TEXT,
    end_time TEXT,
    duration NUMERIC,
    assessment_id TEXT,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id),
    FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id)
);

CREATE TABLE quiz_questions (
    question_id TEXT PRIMARY KEY,
    question_type TEXT,
    question_weight NUMERIC,
    question_due TEXT
);

CREATE TABLE courses (
    course_id TEXT PRIMARY KEY,
    course_name TEXT,
    start_time TEXT,
    end_time TEXT
);

CREATE TABLE course_learner (
    hash_id VARCHAR(255) PRIMARY KEY,
    final_grade NUMERIC,
    enrollment_mode TEXT,
    certificate_status TEXT,
    register_time TEXT,
    group_type TEXT,
    group_name TEXT,
    segment TEXT,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE learner_demographic (
    hash_id VARCHAR(255) PRIMARY KEY,
    gender TEXT,
    year_of_birth INTEGER,
    level_of_education TEXT,
    country TEXT,
    email TEXT,
    segment TEXT,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE course_elements (
    element_id TEXT PRIMARY KEY,
    element_type TEXT,
    week INTEGER,
    course_id TEXT,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

CREATE TABLE learner_index (
    global_learner_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    hash_id VARCHAR(255) NOT NULL,
    FOREIGN KEY (course_id) REFERENCES courses(course_id),
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE metadata (
    course_id TEXT PRIMARY KEY,
    object TEXT,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

CREATE TABLE forum_interaction (
    post_id TEXT PRIMARY KEY,
    hash_id VARCHAR(255),
    post_type TEXT,
    post_title TEXT,
    post_content TEXT,
    post_timestamp TEXT,
    post_parent_id TEXT,
    post_thread_id TEXT,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id)
);

CREATE TABLE survey_descriptions (
    question_id TEXT PRIMARY KEY,
    course_id TEXT,
    question_type TEXT,
    question_description TEXT,
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

CREATE TABLE survey_responses (
    response_id TEXT PRIMARY KEY,
    hash_id VARCHAR(255),
    question_id TEXT,
    answer TEXT,
    FOREIGN KEY (hash_id) REFERENCES user_profiles(hash_id),
    FOREIGN KEY (question_id) REFERENCES survey_descriptions(question_id)
);
