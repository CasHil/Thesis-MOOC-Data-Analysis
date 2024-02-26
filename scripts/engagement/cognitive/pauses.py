import sqlite3
import glob 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

WORKING_DIR = 'W:/staff-umbrella/gdicsmoocs/Working copy'
DB_LOCATION = WORKING_DIR + '/scripts/thesis_db'
COURSES = ['EX101x', 'ST1x', 'UnixTx']

def find_all_log_files_for_course_id(course_id):
    return glob.glob(f"{WORKING_DIR}/{course_id}*/*.log", recursive=True)

def find_presurvey_files_for_course_id(course_id):
    return glob.glob(f"{WORKING_DIR}/{course_id}*/pre_survey*.txt", recursive=True)

def find_all_pause_events_for_course_id(course_id):
    pause_events = []
    for log_file in find_all_log_files_for_course_id(course_id):
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'pause_video' in line:
                    pause_events.append(line)
    return pause_events

# def find_all_pause_events_for_course_id_limit_5(course_id):
#     pause_events = []
#     for i, log_file in enumerate(find_all_log_files_for_course_id(course_id)):
#         if i == 5:
#             break
#         with open(log_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if 'pause_video' in line:
#                     pause_events.append(line)
#     return pause_events

def find_gender_by_user_id(user_id):
    con = sqlite3.connect(DB_LOCATION)
    cur = con.cursor()
    query = 'SELECT gender FROM user_profiles WHERE hash_id = ?'
    cur.execute(query, (user_id,))
    result = cur.fetchone()
    if result:
        return result[0]
    return None

def calculate_pause_counts(course_id):
    con = sqlite3.connect(DB_LOCATION)
    cur = con.cursor()
    pause_events = find_all_pause_events_for_course_id_limit_5(course_id)

    user_pause_counts = {}
    for pause_event in pause_events:
        if '"user_id":null' in pause_event:
            continue
        user_id = pause_event.split('"user_id":"')[1].split('"')[0]
        if user_id in user_pause_counts:
            user_pause_counts[user_id] += 1
        else:
            user_pause_counts[user_id] = 1

    gender_pause_counts = {'m': [], 'f': [], 'o': [], None: []}
    for user_id, pause_count in user_pause_counts.items():
            gender = find_gender_by_user_id(user_id)
            gender_pause_counts[gender].append(pause_count)

    gender_pause_averages = {}
    for gender, pause_counts in gender_pause_counts.items():
        if pause_counts:
            average = sum(pause_counts) / len(pause_counts)
            gender_pause_averages[gender] = average
        
    cur.close()
    con.close()

    return gender_pause_counts, gender_pause_averages, user_pause_counts

def graph_average_pauses_per_gender(average_pauses_df):
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette(["#fe9929", "#1f78b4", "#f768a1", "#33a02c"])
    sns.set_palette(colors)
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    g = sns.catplot(x='Course', y='Value', hue='Gender', kind='bar', data=average_pauses_df, height=6, aspect=1.5)
    plt.xticks(rotation=45)
    plt.ylabel('Average Pauses')
    plt.xlabel('Course')
    plt.title('Average Pauses by Gender across Courses')

    plt.margins(0.1)

    g.savefig('graphs/average_pauses_per_gender.png')

def create_average_pause_df(pauses_per_course):
    rows = []
    for course, genders in pauses_per_course.items():
        for gender, value in genders.items():
            rows.append({"Course": course, "Gender": gender, "Value": value})

    df = pd.DataFrame(rows)
    gender_map = {'m': 'Male', 'f': 'Female', 'o': 'Other', None: 'Prefer not to say / Unknown'}
    df['Gender'] = df['Gender'].map(gender_map)
    return df

def create_pause_count_df(pause_counts, course_id):
    rows = []
    for gender, counts in pause_counts.items():
        for count in counts:
            rows.append({"Course": course_id, "Gender": gender, "PauseCount": count})

    df = pd.DataFrame(rows)
    gender_map = {'m': 'Male', 'f': 'Female', 'o': 'Other', None: 'Prefer not to say / Unknown'}
    df['Gender'] = df['Gender'].map(gender_map)
    return df

def create_user_id_pause_count(pause_counts, course_id):
    rows = []
    for user, count in pause_counts.items():
        gender = find_gender_by_user_id(user)
        rows.append({"Course": course_id, "User": user, "Gender": gender, "PauseCount": count})
    df = pd.DataFrame(rows)
    
    gender_map = {'m': 'Male', 'f': 'Female', 'o': 'Other', None: 'Prefer not to say / Unknown'}
    df['Gender'] = df['Gender'].map(gender_map)
    
    pre_survey_accumulated = pd.DataFrame()

    for pre_survey_file in find_presurvey_files_for_course_id(course_id):
        pre_survey = pd.read_csv(pre_survey_file, sep='\t')
        pre_survey = pre_survey[['hash_id', 'Q108_1']].dropna()
        
        pre_survey_accumulated = pd.concat([pre_survey_accumulated, pre_survey], ignore_index=True)
    
    pre_survey_accumulated = pre_survey_accumulated.drop_duplicates(subset=['hash_id'])
    
    df = df.merge(pre_survey_accumulated, left_on='User', right_on='hash_id', how='inner')
    return df

def test_correlation_between_video_preferences_and_pauses(df):
    POSSIBLE_ANSWERS = {
        'Not at all important': 1,
        'Slightly important': 2,
        'Moderately important': 3,
        'Important': 4,
        'Very important': 5,
        'Extremely important': 6
    }
    
    df['Q108_1'] = df['Q108_1'].astype(str)
    df['Q108_1_numeric'] = df['Q108_1'].map(POSSIBLE_ANSWERS)
    
    correlation, p_value = spearmanr(df['Q108_1_numeric'], df['PauseCount'])
    
    return correlation, p_value

def regression_with_interaction(df):
    df['Q108_1_numeric'] = df['Q108_1'].map({
        'Not at all important': 1,
        'Slightly important': 2,
        'Moderately important': 3,
        'Important': 4,
        'Very important': 5,
        'Extremely important': 6
    })
    
    if df['Gender'].dtype == 'object':
        df['Gender_code'] = df['Gender'].astype('category').cat.codes
    else:
        df['Gender_code'] = df['Gender']
    
    # Fit the regression model with an interaction term between Q108_1_numeric and Gender
    model = smf.ols('PauseCount ~ Q108_1_numeric * Gender_code', data=df).fit()
    return model.summary()

def descriptive_statistics(df):
    desc_stats = df.groupby(['Gender'])['PauseCount'].describe(percentiles=[.25, .5, .75])
    desc_stats['IQR'] = desc_stats['75%'] - desc_stats['25%']
    desc_stats['Variance'] = df.groupby(['Gender'])['PauseCount'].var() 
    return desc_stats

def test_normality(df, value_col):
    p_values = {}
    for group in df['Gender'].unique():
        group_df = df[df['Gender'] == group][value_col]
        if len(group_df) < 3:
            continue
        stat, p = shapiro(group_df)
        p_values[group] = p
    return p_values

def test_homogeneity_of_variances(df, value_col):
    samples = [group[value_col].values for name, group in df.groupby('Gender')]
    stat, p = levene(*samples)
    return p

def perform_ttest(df, value_col, group1_label, group2_label):
    group1_values = df[df['Gender'] == group1_label][value_col]
    group2_values = df[df['Gender'] == group2_label][value_col]
    t_stat, p_value = ttest_ind(group1_values, group2_values, equal_var=True)
    return p_value

def perform_mannwhitney(df, value_col, group1_label, group2_label):
    group1_values = df[df['Gender'] == group1_label][value_col]
    group2_values = df[df['Gender'] == group2_label][value_col]
    u_stat, p_value = mannwhitneyu(group1_values, group2_values)
    return p_value

def main():
    average_pauses_per_course = {}
    gender_pause_counts_per_course = {}

    for course_id in COURSES:
        print(f"Analysing pauses in course: {course_id}")
        gender_pause_counts, gender_pause_averages, user_pause_counts = calculate_pause_counts(course_id)
        gender_pause_counts_per_course[course_id] = gender_pause_counts
        average_pauses_per_course[course_id] = gender_pause_averages
        
        user_pause_df = create_user_id_pause_count(user_pause_counts, course_id)
        correlation, p_value = test_correlation_between_video_preferences_and_pauses(user_pause_df)
        print("Correlation between video preferences and pauses:", correlation)
        print("P-Value:", p_value)
        print("---------------------------------------------")
        print("Regression with Interaction Results:")
        print(regression_with_interaction(user_pause_df))
        print("---------------------------------------------")
        pause_count_df = create_pause_count_df(gender_pause_counts_per_course[course_id], course_id)
        pause_count_df = pause_count_df.loc[pause_count_df['Gender'].isin(['Male', 'Female'])]
        stats = descriptive_statistics(pause_count_df)
        print(stats)
        print("Normality Test Results:", test_normality(pause_count_df, 'PauseCount'))
        print("Homogeneity of Variances Test P-Value:", test_homogeneity_of_variances(pause_count_df, 'PauseCount'))
        print("T-Test P-Value between Male and Female:", perform_ttest(pause_count_df, 'PauseCount', 'Male', 'Female'))
        print("Mann-Whitney U Test P-Value between Male and Female:", perform_mannwhitney(pause_count_df, 'PauseCount', 'Male', 'Female'))
        print("=============================================")
    average_pause_df = create_average_pause_df(average_pauses_per_course)
    graph_average_pauses_per_gender(average_pause_df)



if __name__ == '__main__':
    main()