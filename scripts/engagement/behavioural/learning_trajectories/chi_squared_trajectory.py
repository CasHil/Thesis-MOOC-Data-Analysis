import pandas as pd
import scipy.stats as stats
import glob
from dotenv import load_dotenv
import os
import json

load_dotenv()
COURSES = json.loads(os.getenv("COURSES"))

def find_course_cluster_data(course: str) -> list[str]:
    files = glob.glob(f'./output/**/{course}**.csv')
    if len(files) == 0:
        print(f'No data found for {course}')
        return []
    return files

def main():
    # Load data
    for COURSE in COURSES:
        auditing = 0
        completing  = 0
        disenngaging = 0
        sampling = 0

        files = find_course_cluster_data(COURSE)
        for file in files:
            for line in open(file, 'r'):
                cluster = line.split(',')[0]
                number = int(line.split(',')[1])
                if cluster == 'auditing':
                    auditing += number
                elif cluster == 'completing':
                    completing += number
                elif cluster == 'disengaging':
                    disenngaging += number
                elif cluster == 'sampling':
                    sampling += number
            
    
    

if __name__ == '__main__':
    main()