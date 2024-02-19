OUTPUT_FOLDER = './insert_scripts/'

def insert_certificates():
    certificate_files = find_certificate_files()

    with open(OUTPUT_FOLDER + 'insert_certificates.sql', 'w') as output:
        for certificate_file in certificate_files:
            with open(certificate_file, 'r') as f:
                output.write('INSERT INTO public.certificates (hash_id, grade, course_id, status, created_date, modified_date, mode) VALUES\n')
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
                        
                        output.write("('{}', '{}', '{}', '{}', '{}', '{}', '{}'),\n".format(data[0], data[1], data[2], data[3], data[4], data[5], data[6]))                    
            # Remove last trailing comma
            output.seek(output.tell() - 3)
            output.write("")
            output.write("\n")
            output.write("ON CONFLICT DO NOTHING;\n\n")
    
    with open(OUTPUT_FOLDER + 'insert_certificates.sql', 'r+') as f:
        data = f.read()
        data = data.replace("'NULL'", "NULL")
        f.seek(0)
        f.write(data)
        f.truncate()

def find_certificate_files():
    import os
    certificate_files = []
    for root, _, files in os.walk("G:/staff-umbrella/gdicsmoocs/Working copy"):
        for file in files:
            if file.endswith("certificates_generatedcertificate-prod-analytics.sql"):
                certificate_files.append(os.path.join(root, file))
    return certificate_files
def main():
    insert_certificates()

if __name__ == '__main__':
    main()