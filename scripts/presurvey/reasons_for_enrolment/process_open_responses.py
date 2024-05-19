import glob
import pandas as pd
import os
import nltk

from nltk.corpus import stopwords
from string import punctuation

nltk.download('punkt')
nltk.download('stopwords')


def main():
    open_response_files = glob.glob('open_responses/*.csv')
    open_responses_dir = 'open_responses'
    automated_classification_dir = os.path.join(
        open_responses_dir, 'classification', 'automated')
    automated_counts_dir = os.path.join(
        open_responses_dir, 'counts', 'automated')

    os.makedirs(automated_classification_dir, exist_ok=True)
    os.makedirs(automated_counts_dir, exist_ok=True)

    understanding_and_learning_cluster_name = 'Understanding and Learning'
    interested_cluster_name = 'Interested'
    studies_cluster_name = 'Studies'
    career_cluster_name = 'Career'
    other_cluster_name = 'Other'
    teaching_cluster_name = 'Teaching'

    for file in open_response_files:
        course = file.split(os.sep)[-1].split('_')[0]
        counts = pd.DataFrame(columns=[
                              'Gender', understanding_and_learning_cluster_name, interested_cluster_name, studies_cluster_name, career_cluster_name, other_cluster_name, teaching_cluster_name])
        df = pd.read_csv(file)

        # Create a DataFrame for classified responses
        classified_responses = pd.DataFrame(
            columns=['Response', 'Classification'])

        teacher_words = ['teach', 'instruct',
                         'professor', 'lecturer']
        understanding_and_learning_words = ['learn', 'educat', 'knowledg',
                                            'understand', 'inform']
        interest_words = ['interest', 'curio',
                          'fascin', 'passion', 'hobb']
        studies_words = ['study', 'studi', 'degree', 'program', 'course', 'subject', 'major', 'minor', 'college',
                         'univ', 'hs', 'school', 'master', 'bachelor', 'phd', 'certif', 'diploma', 'academ', 'homework', 'class', 'grad']
        career_words = ['career', 'work', 'job', 'business', 'tester', 'developer', 'programmer', 'engineer', 'analyst', 'designer',
                        'manager', 'consultant', 'entrepreneur', 'QA', 'roles', 'sysadmin', 'position', 'field', 'opportun', 'profess', 'intern']

        for _, row in df.iterrows():
            response: str = row.iloc[0].lower()
            response_without_punctuation = ''.join(
                [char for char in response if char not in punctuation])
            response_without_stop_words = ' '.join([word for word in response_without_punctuation.split()
                                                    if word not in stopwords.words('english')])
            response_words = response_without_stop_words.split()

            word_counts = {
                teaching_cluster_name: 0,
                understanding_and_learning_cluster_name: 0,
                interested_cluster_name: 0,
                studies_cluster_name: 0,
                career_cluster_name: 0,
                other_cluster_name: 0
            }

            gender = row.iloc[1]

            for word in response_words:
                if any(substring in word for substring in teacher_words):
                    word_counts[teaching_cluster_name] += 1

                if any(substring in word for substring in understanding_and_learning_words):
                    word_counts[understanding_and_learning_cluster_name] += 1

                if any(substring in word for substring in interest_words):
                    word_counts[interested_cluster_name] += 1

                if any(substring in word for substring in studies_words):
                    word_counts[studies_cluster_name] += 1

                if any(substring in word for substring in career_words):
                    word_counts[career_cluster_name] += 1

            # 'this class', 'this course', 'the class' and 'the course' should not count as learning
            word_counts[understanding_and_learning_cluster_name] -= response_words.count(
                'this course')
            word_counts[understanding_and_learning_cluster_name] -= response_words.count(
                'this class')
            word_counts[understanding_and_learning_cluster_name] -= response_words.count(
                'the course')
            word_counts[understanding_and_learning_cluster_name] -= response_words.count(
                'the class')

            word_counts[interested_cluster_name] += response_words.count('fun')
            word_counts[career_cluster_name] += response.count(
                'data scientist')

            # If no word group was matched, increment the count for 'Other'
            if all(word_counts[key] == 0 for key in word_counts):
                word_counts[other_cluster_name] += 1

            max_count = max(word_counts.values())

            # If there are multiple clusters with the same count and they are both the maximum, classify as 'Other'
            max_clusters = [k for k, v in word_counts.items()
                            if v == max_count]

            if len(max_clusters) > 1 or max_count == 0:
                # An analysis by hand showed that when there are multiple clusters and one of the max clusters in career, it should usually be career
                if career_cluster_name in max_clusters:
                    cluster = career_cluster_name
                else:
                    cluster = other_cluster_name
            else:
                cluster = max_clusters[0]

            classified_responses = pd.concat([classified_responses, pd.DataFrame(
                {'Response': [response], 'Classification': [cluster]})], ignore_index=True)

            if gender not in counts['Gender'].values:
                counts = pd.concat([counts, pd.DataFrame({'Gender': [gender], teaching_cluster_name: [0], understanding_and_learning_cluster_name: [
                                    0], interested_cluster_name: [0], studies_cluster_name: [0], career_cluster_name: [0], other_cluster_name: [0]})], ignore_index=True)

            counts.loc[counts['Gender'] == gender,
                       teaching_cluster_name] += word_counts[teaching_cluster_name]
            counts.loc[counts['Gender'] == gender,
                       understanding_and_learning_cluster_name] += word_counts[understanding_and_learning_cluster_name]
            counts.loc[counts['Gender'] == gender,
                       interested_cluster_name] += word_counts[interested_cluster_name]
            counts.loc[counts['Gender'] == gender,
                       studies_cluster_name] += word_counts[studies_cluster_name]
            counts.loc[counts['Gender'] == gender,
                       career_cluster_name] += word_counts[career_cluster_name]
            counts.loc[counts['Gender'] == gender,
                       other_cluster_name] += word_counts[other_cluster_name]

        # Save the classified responses to a CSV file
        classified_responses.to_csv(os.path.join(automated_classification_dir, f"""{
                                    course}_classified.csv"""), index=False)

        # Save the counts DataFrame to a CSV file
        counts.to_csv(os.path.join(
            automated_counts_dir, f"{course}_clusters.csv"), index=False)

    should_count_manual_classifications = True
    if should_count_manual_classifications:
        count_manual_classifications()


def count_manual_classifications():
    manual_classifications_dir = 'open_responses/classification/manual'
    manual_counts_dir = 'open_responses/counts/manual'

    os.makedirs(manual_counts_dir, exist_ok=True)
    os.makedirs(manual_classifications_dir, exist_ok=True)

    manual_classifications_files = glob.glob(
        f"{manual_classifications_dir}/*.csv")

    for file in manual_classifications_files:
        df = pd.read_csv(file)

        # Pivot the DataFrame
        pivot_df = df.pivot_table(
            index='gender', columns='classification', aggfunc='size', fill_value=0)

        # Extract the original filename without the extension
        filename = os.path.splitext(os.path.basename(file))[0]

        # Save the pivot DataFrame to a CSV file with the original filename
        pivot_df.to_csv(os.path.join(
            manual_counts_dir, f"{filename}_clusters.csv"))


if __name__ == "__main__":
    main()