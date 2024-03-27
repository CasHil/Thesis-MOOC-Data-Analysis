import os
from dotenv import load_dotenv

load_dotenv()

WORKING_DIRECTORY = os.getenv('WORKING_DIRECTORY')
SCRIPTS_DIRECTORY = os.getenv('SCRIPTS_DIRECTORY')

def insert_certificates():
    certificate_files = find_certificate_files()

    with open(SCRIPTS_DIRECTORY + 'insert_certificates.sql', 'w', encoding='utf-8') as output:
        for certificate_file in certificate_files:
            with open(certificate_file, 'r', encoding='utf-8') as f:
                output.write('INSERT INTO certificates (hash_id, grade, course_id, status, created_date, modified_date, mode) VALUES\n')
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
    
    with open(SCRIPTS_DIRECTORY + 'insert_certificates.sql', 'r+', encoding='utf-8') as f:
        data = f.read()
        data = data.replace("'NULL'", "NULL")
        f.seek(0)
        f.write(data)
        f.truncate()

def find_certificate_files():
    import os
    certificate_files = []
    for root, _, files in os.walk(WORKING_DIRECTORY):
        for file in files:
            if file.endswith("certificates_generatedcertificate-prod-analytics.sql"):
                certificate_files.append(os.path.join(root, file))
    return certificate_files

def main():
    insert_certificates()

if __name__ == '__main__':
    main()