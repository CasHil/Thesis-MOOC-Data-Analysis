import glob
import pandas as pd
import os
import nltk

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


def main():
    open_response_files = glob.glob('open_responses/*.csv')
    open_responses_dir = 'open_responses'
    classification_dir = os.path.join(open_responses_dir, 'classification')
    counts_dir = os.path.join(open_responses_dir, 'counts')

    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(counts_dir, exist_ok=True)

    for file in open_response_files:
        course = file.split(os.sep)[-1].split('_')[0]
        counts = pd.DataFrame(columns=[
                              'Gender', 'Teaching', 'Learning', 'Interested', 'Studies', 'Career', 'Other'])
        df = pd.read_csv(file)

        # Create a DataFrame for classified responses
        classified_responses = pd.DataFrame(
            columns=['Response', 'Classification'])

        teacher_words = ['teach', 'instruct',
                         'professor', 'lecturer']
        learn_words = ['learn', 'educat', 'knowledg',
                       'understand', 'skill', 'inform', 'explore']
        interest_words = ['interest', 'curio',
                          'fascin', 'passion', 'hobb']
        studies_words = ['learn', 'educat', 'knowledg',
                         'understand', 'skill', 'inform', 'explore', 'study', 'studi', 'degree', 'program', 'course', 'subject', 'major', 'minor', 'college',
                         'univ', 'hs', 'school', 'master', 'bachelor', 'phd', 'certif', 'diploma', 'academ', 'homework', 'class', 'grad']
        career_words = ['career', 'work', 'job', 'business', 'tester', 'developer', 'programmer', 'engineer', 'analyst', 'designer',
                        'manager', 'consultant', 'entrepreneur', 'QA', 'roles', 'sysadmin', 'position', 'data scientist', 'field', 'opportun', 'profess']

        for _, row in df.iterrows():
            response = row.iloc[0].lower()
            response_without_stop_words = ' '.join([word for word in response.split()
                                                    if word not in stopwords.words('english')])
            response_words = response_without_stop_words.split()

            word_counts = {
                'Teaching': 0,
                'Learning': 0,
                'Interested': 0,
                'Studies': 0,
                'Career': 0,
                'Other': 0
            }

            gender = row.iloc[1]  # Use iloc for positional indexing

            for word in response_words:
                if any(substring in word for substring in teacher_words):
                    word_counts['Teaching'] += 1

                if any(substring in word for substring in learn_words):
                    word_counts['Learning'] += 1

                if any(substring in word for substring in interest_words):
                    word_counts['Interested'] += 1

                if any(substring in word for substring in studies_words):
                    word_counts['Studies'] += 1

                if any(substring in word for substring in career_words):
                    word_counts['Career'] += 1

            # 'this class', 'this course', 'the class' and 'the course' should not count as learning
            word_counts['Learning'] -= response_words.count('this course')
            word_counts['Learning'] -= response_words.count('this class')
            word_counts['Learning'] -= response_words.count('the course')
            word_counts['Learning'] -= response_words.count('the class')

            # If no word group was matched, increment the count for 'Other'
            if all(word_counts[key] == 0 for key in word_counts):
                word_counts['Other'] += 1

            max_count = max(word_counts.values())

            # If there are multiple clusters with the same count and they are both the maximum, classify as 'Other'
            max_clusters = [k for k, v in word_counts.items()
                            if v == max_count]

            if len(max_clusters) > 1 or max_count == 0:
                cluster = 'Other'
            else:
                cluster = max_clusters[0]

            classified_responses = pd.concat([classified_responses, pd.DataFrame(
                {'Response': [response], 'Classification': [cluster]})], ignore_index=True)

            if gender not in counts['Gender'].values:
                counts = pd.concat([counts, pd.DataFrame({'Gender': [gender], 'Teaching': [0], 'Learning': [
                                    0], 'Interested': [0], 'Studies': [0], 'Career': [0], 'Other': [0]})], ignore_index=True)

            counts.loc[counts['Gender'] == gender,
                       'Teaching'] += word_counts['Teaching']
            counts.loc[counts['Gender'] == gender,
                       'Learning'] += word_counts['Learning']
            counts.loc[counts['Gender'] == gender,
                       'Interested'] += word_counts['Interested']
            counts.loc[counts['Gender'] == gender,
                       'Studies'] += word_counts['Studies']
            counts.loc[counts['Gender'] == gender,
                       'Career'] += word_counts['Career']
            counts.loc[counts['Gender'] == gender,
                       'Other'] += word_counts['Other']

        # Save the classified responses to a CSV file
        classified_responses.to_csv(os.path.join(classification_dir, f"""{
                                    course}_classified.csv"""), index=False)

        # Save the counts DataFrame to a CSV file
        counts.to_csv(os.path.join(
            counts_dir, f"{course}_clusters.csv"), index=False)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
