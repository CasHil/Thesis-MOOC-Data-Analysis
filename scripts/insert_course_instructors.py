OUTPUT_FOLDER = './insert_scripts/'

def insert_course_instructors():
    course_instructor_files = find_course_instructor_files()
    with open(OUTPUT_FOLDER + 'insert_course_instructors.sql', 'w') as output:
        for course_instructor_file in course_instructor_files:
            with open(course_instructor_file, 'r') as f:
                output.write('INSERT INTO public.course_instructors (hash_id, org, course_id, role) VALUES\n')
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
                            # Surround all strings with single quotes if they are not yet surrounded by single quotes
                        
                        output.write("('{}', '{}', '{}', '{}'),\n".format(data[0], data[1], data[2], data[3]))                    
                # Remove last trailing comma
            output.seek(output.tell() - 3)
            output.write("")
            output.write("\n")
            output.write("ON CONFLICT DO NOTHING;\n\n")
    
    with open(OUTPUT_FOLDER + 'insert_course_instructors.sql', 'r+') as f:
        data = f.read()
        data = data.replace("'NULL'", "NULL")
        f.seek(0)
        f.write(data)
        f.truncate()

def find_course_instructor_files():
    import os
    course_instructor_files = []
    for root, _, files in os.walk("G:/staff-umbrella/gdicsmoocs/Working copy"):
        for file in files:
            if file.endswith("student_courseaccessrole-prod-analytics.sql"):
                course_instructor_files.append(os.path.join(root, file))
    return course_instructor_files

def main():
    insert_course_instructors()

if __name__ == '__main__':
    main()