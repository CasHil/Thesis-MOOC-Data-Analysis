from typing import List
import pandas as pd
from scipy.stats import chi2_contingency
import glob
from dotenv import load_dotenv
import os
import json
import numpy as np

load_dotenv()
COURSES: List[str] = json.loads(os.getenv("COURSES"))

def find_course_cluster_data(course: str) -> List[str]:
    files: List[str] = glob.glob(f'**/*{course}*_response_cluster_counts.csv', recursive=True)
    if len(files) == 0:
        print(f'No data found for {course}')
        return []
    return files

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def main() -> None:
    # Initialize DataFrame to store counts for each course
    course_counts: pd.DataFrame = pd.DataFrame()

    # Load data
    for COURSE in COURSES:
        course_df: pd.DataFrame = pd.DataFrame()

        files: List[str] = find_course_cluster_data(COURSE)

        # For each file, read the data into a DataFrame and sum the counts for each cluster
        for file in files:
            df: pd.DataFrame = pd.read_csv(file)

            # Check if 'Female' or 'Male' is in the filename and add a 'gender' column
            if 'Female' in file:
                df['gender'] = 'Female'
            elif 'Male' in file:
                df['gender'] = 'Male'
            else:
                df['gender'] = 'Unknown'

            # Ensure the engagement clusters are present as columns
            engagement_clusters = ['Auditing', 'Completing', 'Disengaging', 'Sampling']
            for cluster in engagement_clusters:
                if cluster not in df.columns:
                    df[cluster] = 0

            course_df = pd.concat([course_df, df])

        # Group by response and gender and sum the counts for each cluster
        course_df = course_df.groupby(['response', 'gender']).sum().reset_index()

        # Add a column for the course
        course_df['course'] = COURSE

        # Append the counts for this course to the course_counts DataFrame
        course_counts = pd.concat([course_counts, course_df])

    # Reset the index of the final DataFrame
    course_counts.reset_index(drop=True, inplace=True)

    female_counts_chi2 = course_counts[course_counts['gender'] == 'Female'].drop(columns=['gender', 'course', 'response'])
    male_counts_chi2 = course_counts[course_counts['gender'] == 'Male'].drop(columns=['gender', 'course', 'response'])

    chi2_female, p_female, dof_female, expected_female = chi2_contingency(female_counts_chi2)
    chi2_male, p_male, dof_male, expected_male = chi2_contingency(male_counts_chi2)

    # For final results, include 'response'
    female_counts_final = course_counts[course_counts['gender'] == 'Female'].drop(columns=['gender', 'course'])
    male_counts_final = course_counts[course_counts['gender'] == 'Male'].drop(columns=['gender', 'course'])

    grouped_female_counts_final = female_counts_final.groupby('response').sum()
    grouped_male_counts_final = male_counts_final.groupby('response').sum()

    with open('contingency_tables.txt', 'w') as f:
        # Write the overall contingency tables and chi2, p statistics
        f.write("Overall:\n")
        f.write(f"Female - Chi2: {chi2_female}, p-value: {p_female}\n")
        f.write(grouped_female_counts_final.to_string())
        f.write("\n")
        f.write(f"Male - Chi2: {chi2_male}, p-value: {p_male}\n")
        f.write(grouped_male_counts_final.to_string())
        f.write("\n")

        # Write the contingency tables and chi2, p statistics for each course
        for course in COURSES:
            female_counts_course_chi2 = course_counts[(course_counts['course'] == course) & (course_counts['gender'] == 'Female')].drop(columns=['gender', 'course', 'response'])
            male_counts_course_chi2 = course_counts[(course_counts['course'] == course) & (course_counts['gender'] == 'Male')].drop(columns=['gender', 'course', 'response'])

            chi2_female, p_female, dof_female, expected_female = chi2_contingency(female_counts_course_chi2)
            chi2_male, p_male, dof_male, expected_male = chi2_contingency(male_counts_course_chi2)

            female_counts_course_final = course_counts[(course_counts['course'] == course) & (course_counts['gender'] == 'Female')].drop(columns=['gender', 'course'])
            male_counts_course_final = course_counts[(course_counts['course'] == course) & (course_counts['gender'] == 'Male')].drop(columns=['gender', 'course'])

            f.write(f"For course {course}:\n")
            f.write(f"Female - Chi2: {chi2_female}, p-value: {p_female}\n")
            f.write(female_counts_course_final.to_string())
            f.write("\n")
            f.write(f"Male - Chi2: {chi2_male}, p-value: {p_male}\n")
            f.write(male_counts_course_final.to_string())
            f.write("\n")

if __name__ == "__main__":
    main()
