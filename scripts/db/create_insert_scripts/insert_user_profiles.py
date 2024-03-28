import os
from dotenv import load_dotenv

load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
SCRIPTS_DIRECTORY = os.getenv('SCRIPTS_DIRECTORY')


def insert_user_profiles() -> None:
    user_profile_files = find_user_profile_files()

    with open(SCRIPTS_DIRECTORY + 'insert_user_profiles.sql', 'w', encoding='utf-8') as output:
        for user_profile_file in user_profile_files:
            with open(user_profile_file, 'r', encoding='utf-8') as f:
                output.write(
                    'INSERT INTO user_profiles (hash_id, language, gender, year_of_birth, level_of_education, goals, country) VALUES\n')
                for index, line in enumerate(f):
                    if index == 0:
                        continue
                    line = line.strip()
                    if line:
                        data = line.split('\t')
                        while len(data) < 7:
                            data.append('NULL')
                        # Replace any empty strings with NULL
                        for i in range(len(data)):
                            if data[i] == '':
                                data[i] = 'NULL'
                            if '\'' in data[i]:
                                data[i] = data[i].replace('\'', '\'\'')
                        output.write("('{}', '{}', '{}', '{}', '{}', '{}', '{}'),\n".format(
                            data[0], data[1], data[2], data[3], data[4], data[5], data[6]))
            # Remove last trailing comma
            output.seek(output.tell() - 3)
            output.write("")
            output.write("\n")
            output.write("ON CONFLICT DO NOTHING;\n\n")

    with open(SCRIPTS_DIRECTORY + 'insert_user_profiles.sql', 'r+', encoding='utf-8') as f:
        data = f.read()
        data = data.replace("'NULL'", "NULL")
        f.seek(0)
        f.write(data)
        f.truncate()


def find_user_profile_files() -> list[str]:
    import os
    user_profile_files = []
    for root, _, files in os.walk(WORKING_DIRECTORY):
        for file in files:
            if file.endswith("auth_userprofile-prod-analytics.sql"):
                user_profile_files.append(os.path.join(root, file))
    return user_profile_files


def main() -> None:
    insert_user_profiles()


if __name__ == '__main__':
    main()
