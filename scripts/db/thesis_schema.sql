CREATE TABLE certificates (
    hash_id VARCHAR(255) NOT NULL,
    grade NUMERIC(5,2),
    course_id VARCHAR(255) NOT NULL,
    status VARCHAR(50),
    created_date DATETIME NOT NULL,
    modified_date DATETIME NOT NULL,
    mode VARCHAR(50),
    PRIMARY KEY (hash_id, course_id)
);

CREATE TABLE course_instructors (
    hash_id VARCHAR(255) NOT NULL,
    org VARCHAR(255) NOT NULL,
    course_id VARCHAR(255) NOT NULL,
    role VARCHAR(100) NOT NULL,
    PRIMARY KEY (hash_id, course_id)
);

CREATE TABLE enrollments (
    hash_id VARCHAR(255) NOT NULL,
    course_id VARCHAR(255) NOT NULL,
    created DATETIME NOT NULL,
    is_active BOOLEAN NOT NULL,
    mode VARCHAR(50) NOT NULL,
    PRIMARY KEY (hash_id, course_id)
);

CREATE TABLE grades (
    hash_id VARCHAR(255) NOT NULL,
    course_id VARCHAR(255) NOT NULL,
    percent_grade NUMERIC(5,2),
    letter_grade VARCHAR(100),
    passed_timestamp DATETIME,
    created DATETIME NOT NULL,
    modified DATETIME NOT NULL,
    PRIMARY KEY (hash_id, course_id)
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