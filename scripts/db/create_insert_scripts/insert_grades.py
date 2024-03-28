import os
from dotenv import load_dotenv

load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
SCRIPTS_DIRECTORY = os.getenv('SCRIPTS_DIRECTORY')


def insert_grades() -> None:
    grade_files = find_grade_files()

    with open(SCRIPTS_DIRECTORY + 'insert_grades.sql', 'w', encoding='utf-8') as output:
        for grade_file in grade_files:
            with open(grade_file, 'r', encoding='utf-8') as f:
                output.write(
                    'INSERT INTO grades (hash_id, course_id, percent_grade, letter_grade, passed_timestamp, created, modified) VALUES\n')
                for index, line in enumerate(f):
                    if index == 0:
                        continue
                    line = line.strip()
                    if line:
                        data = line.split('\t')
                        # Replace any empty strings with NULL
                        for i in range(len(data)):
                            if data[i] == '':
                                data[i] = 'NULL'

                        output.write("('{}', '{}', '{}', '{}', '{}', '{}', '{}'),\n".format(
                            data[0], data[1], data[2], data[3], data[4], data[5], data[6]))
            # Remove last trailing comma
            output.seek(output.tell() - 3)
            output.write("")
            output.write("\n")
            output.write("ON CONFLICT DO NOTHING;\n\n")

    with open(SCRIPTS_DIRECTORY + 'insert_grades.sql', 'r+', encoding='utf-8') as f:
        data = f.read()
        data = data.replace("'NULL'", "NULL")
        f.seek(0)
        f.write(data)
        f.truncate()


def find_grade_files() -> list[str]:
    import os
    grade_files = []
    for root, _, files in os.walk(WORKING_DIRECTORY):
        for file in files:
            if file.endswith("grades_persistentcoursegrade-prod-analytics.sql"):
                grade_files.append(os.path.join(root, file))
    return grade_files


def main():
    insert_grades()


if __name__ == '__main__':
    main()
