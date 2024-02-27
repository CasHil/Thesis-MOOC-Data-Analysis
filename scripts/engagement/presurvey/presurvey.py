import pandas as pd

WORKING_DIR = 'W:/staff-umbrella/gdicsmoocs/Working copy'
PRESURVEY = WORKING_DIR + '/EX101x_2T2018_run6/pre_survey_results_EX101x_2T2018.txt'

def main():
    # Read the presurvey file. It is a TSV but in a txt format. Print the first 10 lines in table format.
    presurvey = pd.read_csv(PRESURVEY, sep='\t')
    # I want to see Q108
    print(presurvey['Q108_1'].value_counts())
    # Drop StartDate and EndDate
    # presurvey = presurvey.drop(columns=['StartDate', 'EndDate', 'Status', 'Progress', 'Duration..in.seconds.', 'Finished', 'RecordedDate', 'ResponseId', 'DistributionChannel', 'hash_id', 'course_id', 'course_type', 'Q4.2', 'Q4.2.4_1', 'Q4.2.4_2', 'Q4.3_1', 'Q4.4', 'Q4.5_1', 'Q4.5_2', 'Q4.6_9_TEXT', 'SP'])
    # print(presurvey.head(10))

    # Q4.2: Which of the following best describes your current employment status?
    # Q4.3_1: What is your age? - Years:
    # Q4.2.4_1: In which industry do you currently work - Sector
    # Q4.4: What is your gender?
    # Q4.5_1: What is your (first) nationality (continent)
    # Q4.5_2: What is your (first) nationality (continent)?
    # Q4.6_9_TEXT: What is the highest degree or level of education you have completed?

if __name__ == "__main__":
    main()