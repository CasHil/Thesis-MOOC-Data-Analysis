import pandas as pd
import os

def save_df_as_int(df: pd.DataFrame, file_name: str):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].fillna('NaN').astype(str)
    df.to_csv(file_name)


def main():
    male_df = pd.DataFrame()
    female_df = pd.DataFrame()

    for file in os.listdir():
        if file.endswith(".csv"):
            print(file)
            df = pd.read_csv(file)
            file_name = file[:-4]  # Remove .csv from file name

            for gender in ['m', 'f']:
                filtered_df = df[df['gender'] == gender]
                counts = filtered_df['response'].value_counts()
                counts = counts.astype(int)

                if gender == 'm':
                    male_df[file_name] = counts
                else:
                    female_df[file_name] = counts

    save_df_as_int(male_df, 'male.csv')
    save_df_as_int(female_df, 'female.csv')


if __name__ == "__main__":
    main()
