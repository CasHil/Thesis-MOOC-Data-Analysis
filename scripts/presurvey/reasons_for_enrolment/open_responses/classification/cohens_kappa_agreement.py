import os
import json
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def main():
    load_dotenv()
    courses = json.loads(os.getenv("COURSES"))
    automated_classification_dir = os.path.join(os.path.curdir, "automated")
    manual_classification_dir = os.path.join(os.path.curdir, "manual")

    all_automated_df = pd.DataFrame()
    all_manual_df = pd.DataFrame()

    for course in courses:
        automated_file = os.path.join(
            automated_classification_dir, f"{course}_classified.csv")
        manual_file = os.path.join(
            manual_classification_dir, f"{course}_classified.csv")

        automated_df = pd.read_csv(automated_file)
        manual_df = pd.read_csv(manual_file)

        # Only keep classification directory
        automated_df = automated_df[['classification']]
        manual_df = manual_df[['classification']]

        # Calculate Cohen's kappa for the current course
        kappa = cohen_kappa_score(
            automated_df['classification'], manual_df['classification'])
        print(f"Cohen's Kappa for course {course}: {kappa}")

        all_automated_df = pd.concat([all_automated_df, automated_df])
        all_manual_df = pd.concat([all_manual_df, manual_df])

    # Ensure that all_automated_df and all_manual_df have the same number of rows
    assert len(all_automated_df) == len(
        all_manual_df), "DataFrames have different number of rows"

    # Calculate Cohen's kappa for the entire set of classifications
    kappa = cohen_kappa_score(
        all_automated_df['classification'], all_manual_df['classification'])

    # Print the kappa score
    print(f"Cohen's Kappa for all courses: {kappa}")


if __name__ == "__main__":
    main()
