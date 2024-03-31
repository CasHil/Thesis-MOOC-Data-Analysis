import os
from dotenv import load_dotenv

load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
MOOC_DB_DIRECTORY = os.getenv('MOOC_DB_DIRECTORY')


def write_insert_statements(output_file: str, table_name: str, columns: list[str], file_ending: str) -> None:
    target_files = find_files_by_ending(WORKING_DIRECTORY, file_ending)

    with open(os.path.join(os.path.abspath(MOOC_DB_DIRECTORY), output_file), 'w', encoding='utf-8') as output:
        output.write(f"""INSERT INTO {table_name} ({
            ", ".join(columns)}) VALUES\n""")
        lines = []
        for target_file in target_files:
            with open(target_file, 'r', encoding='utf-8') as f:
                for index, line in enumerate(f):
                    if index == 0:
                        continue
                    line_data = line.strip().split('\t')
                    line_data = [data.replace(
                        "'", "''") if data != '' else 'NULL' for data in line_data]
                    while len(line_data) < len(columns):
                        line_data.append('NULL')
                    lines.append(
                        f"({', '.join(f'\'{data}\'' if data != 'NULL' else data for data in line_data)})")
        output.write(",\n".join(lines))
        output.write("\nON CONFLICT DO NOTHING;\n")

    replace_placeholder_with_null(os.path.join(
        os.path.abspath(MOOC_DB_DIRECTORY), output_file))


def find_files_by_ending(search_directory: str, file_ending: str) -> list[str]:
    files_found = []
    for root, _, files in os.walk(search_directory):
        for file in files:
            if file.endswith(file_ending):
                files_found.append(os.path.join(root, file))
    return files_found


def replace_placeholder_with_null(file_path: str) -> None:
    with open(file_path, 'r+', encoding='utf-8') as f:
        data = f.read()
        f.seek(0)
        f.write(data.replace("'NULL'", "NULL"))
        f.truncate()


def main() -> None:
    tables = [
        {
            "table_name": "user_profiles",
            "columns": ["hash_id", "language", "gender", "year_of_birth", "level_of_education", "goals", "country"],
            "file_ending": "auth_userprofile-prod-analytics.sql",
            "output_file": "insert_user_profiles.sql"
        },
        {
            "table_name": "certificates",
            "columns": ["hash_id", "grade", "course_id", "status", "created_date", "modified_date", "mode"],
            "file_ending": "certificates_generatedcertificate-prod-analytics.sql",
            "output_file": "insert_certificates.sql"
        },
        {
            "table_name": "course_instructors",
            "columns": ["hash_id", "org", "course_id", "role"],
            "file_ending": "student_courseaccessrole-prod-analytics.sql",
            "output_file": "insert_course_instructors.sql"
        },
        {
            "table_name": "enrollments",
            "columns": ["hash_id", "course_id", "created", "is_active", "mode"],
            "file_ending": "student_courseenrollment-prod-analytics.sql",
            "output_file": "insert_enrollments.sql"
        },
        {
            "table_name": "grades",
            "columns": ["hash_id", "course_id", "percent_grade", "letter_grade", "passed_timestamp", "created", "modified"],
            "file_ending": "student_courseenrollment-prod-analytics.sql",
            "output_file": "insert_grades.sql"
        }
    ]

    for table in tables:
        write_insert_statements(
            table["output_file"], table["table_name"], table["columns"], table["file_ending"])


if __name__ == '__main__':
    main()
