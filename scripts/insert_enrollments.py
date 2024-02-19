OUTPUT_FOLDER = './insert_scripts/'

def insert_enrollments():
    enrollment_files = find_enrollment_files()

    with open(OUTPUT_FOLDER + 'insert_enrollments.sql', 'w') as output:
        for enrollment_file in enrollment_files:
            with open(enrollment_file, 'r') as f:
                output.write('INSERT INTO public.enrollments (hash_id, course_id, created, is_active, mode) VALUES\n')
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
                        
                        output.write("('{}', '{}', '{}', '{}', '{}'),\n".format(data[0], data[1], data[2], data[3], data[4]))                    
            # Remove last trailing comma
            output.seek(output.tell() - 3)
            output.write("")
            output.write("\n")
            output.write("ON CONFLICT DO NOTHING;\n\n")
    
    with open(OUTPUT_FOLDER + 'insert_enrollments.sql', 'r+') as f:
        data = f.read()
        data = data.replace("'NULL'", "NULL")
        f.seek(0)
        f.write(data)
        f.truncate()

def find_enrollment_files():
    import os
    enrollment_files = []
    for root, _, files in os.walk("G:/staff-umbrella/gdicsmoocs/Working copy"):
        for file in files:
            if file.endswith("student_courseenrollment-prod-analytics.sql"):
                enrollment_files.append(os.path.join(root, file))
    return enrollment_files
def main():
    insert_enrollments()

if __name__ == '__main__':
    main()