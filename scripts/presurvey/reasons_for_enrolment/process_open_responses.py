import glob
import pandas as pd


def main():
    open_response_files = glob.glob('open_responses/*.csv')

    for file in open_response_files:
        counts = pd.DataFrame(columns=[
                              'Gender', 'Teaching', 'Learning', 'Interested', 'Studies', 'Career', 'Other'])

        counts = counts.append('m', (0,) * 6)
        counts = counts.append('f', (0,) * 6)

        df = pd.read_csv(file)
        # print(df.head())
        # Print the first column
        # print(df.iloc[:, 0])

        # Iterate over the first column
        for index, row in df.iterrows():
            response = row[0]
            response = response.lower()
            teacher_words = ['teach', 'instruct', 'professor', 'lecturer']
            learn_words = ['learn',
                           'educat', 'knowledg', 'understand', 'skill', 'inform', 'explore']
            interest_words = ['interest', 'curio', 'fascin', 'passion', 'hobb']
            studies_words = ['study', 'studi', 'degree',
                             'program', 'course', 'subject', 'major', 'minor', 'college', 'univ', 'hs', 'school', 'master', 'bachelor', 'phd', 'certif', 'diploma', 'academ', 'homework', 'class']
            career_words = ['career', 'work', 'job', 'colleague', 'business', 'tester', 'developer',
                            'programmer', 'engineer', 'analyst', 'designer', 'manager', 'consultant', 'entrepreneur', 'roles', 'sysadmin', 'position', 'data scientist', 'field', 'opportun', 'profess']

            gender = row[1]
            if any(word in response for word in teacher_words):

            elif any(word in response for word in learn_words):
                learning += 1
            elif any(word in response for word in interest_words):
                interested += 1
            elif any(word in response for word in studies_words):
                studies += 1
            elif any(word in response for word in career_words):
                career += 1
            else:
                print(response)
                other += 1

            print(f"Teaching: {teaching}")
            print(f"Learning: {learning}")
            print(f"Interested: {interested}")
            print(f"Studies: {studies}")
            print(f"Career: {career}")
            print(f"Other: {other}")

        course = file.split('/')[-1].split('_')[0]
        closed_question_file = glob.glob(
            f"../closed_responses/{course}*.csv")[0]
        closed_question_df = pd.read_csv(closed_question_file)


if __name__ == "__main__":
    main()
