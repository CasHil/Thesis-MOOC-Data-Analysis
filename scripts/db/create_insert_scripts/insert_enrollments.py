import os
from dotenv import load_dotenv

load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
SCRIPTS_DIRECTORY = os.getenv('SCRIPTS_DIRECTORY')


def insert_enrollments() -> None:
    enrollment_files = find_enrollment_files()

    with open(SCRIPTS_DIRECTORY + 'insert_enrollments.sql', 'w', encoding='utf-8') as output:
        for enrollment_file in enrollment_files:
            with open(enrollment_file, 'r', encoding='utf-8') as f:
                output.write(
                    'INSERT INTO enrollments (hash_id, course_id, created, is_active, mode) VALUES\n')
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

                        output.write("('{}', '{}', '{}', '{}', '{}'),\n".format(
                            data[0], data[1], data[2], data[3], data[4]))
            # Remove last trailing comma
            output.seek(output.tell() - 3)
            output.write("")
            output.write("\n")
            output.write("ON CONFLICT DO NOTHING;\n\n")

    with open(SCRIPTS_DIRECTORY + 'insert_enrollments.sql', 'r+', encoding='utf-8') as f:
        data = f.read()
        data = data.replace("'NULL'", "NULL")
        f.seek(0)
        f.write(data)
        f.truncate()


def find_enrollment_files() -> list[str]:
    import os
    enrollment_files = []
    for root, _, files in os.walk(WORKING_DIRECTORY):
        for file in files:
            if file.endswith("student_courseenrollment-prod-analytics.sql"):
                enrollment_files.append(os.path.join(root, file))
    return enrollment_files


def main() -> None:
    insert_enrollments()


if __name__ == '__main__':
    main()
