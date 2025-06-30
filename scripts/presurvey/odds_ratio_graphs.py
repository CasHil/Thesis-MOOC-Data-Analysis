import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os

# Create directory if it doesn't exist
output_dir = './graphs'
os.makedirs(output_dir, exist_ok=True)

# Data from the table for each reason for enrolment and course
data = {
    'EX101x': {'Career': [3301, 1641], 'Interest': [1013, 308], 'Degree': [234, 96], 'Other': [42, 24], 'Teaching': [41, 15], 'Know the Instructor': [10, 2]},
    'FP101x': {'Career': [930, 64], 'Interest': [1263, 91], 'Degree': [71, 9], 'Other': [39, 2], 'Teaching': [0, 0], 'Know the Instructor': [129, 1]},
    'ST1x': {'Career': [1047, 560], 'Interest': [162, 64], 'Degree': [126, 59], 'Other': [7, 6], 'Teaching': [33, 13], 'Know the Instructor': [0, 0]},
    'UnixTx': {'Career': [238, 55], 'Interest': [94, 7], 'Degree': [66, 18], 'Other': [7, 1], 'Teaching': [3, 4], 'Know the Instructor': [0, 0]}
}

# Total number of responses for each course and gender
total_men = {'EX101x': 4641, 'FP101x': 2432, 'ST1x': 1375, 'UnixTx': 408}
total_women = {'EX101x': 2086, 'FP101x': 167, 'ST1x': 702, 'UnixTx': 85}

# Function to safely calculate odds ratios and confidence intervals
def calculate_odds_ratios(men_count, women_count, total_men, total_women):
    if men_count == 0 or women_count == 0 or (total_men - men_count) == 0 or (total_women - women_count) == 0:
        return None, None, None
    odds_men = men_count / (total_men - men_count)
    odds_women = women_count / (total_women - women_count)
    or_value = odds_men / odds_women
    se_log_or = np.sqrt(1/men_count + 1/(total_men - men_count) + 1/women_count + 1/(total_women - women_count))
    ci_lower = np.exp(np.log(or_value) - 1.96 * se_log_or)
    ci_upper = np.exp(np.log(or_value) + 1.96 * se_log_or)
    return or_value, ci_lower, ci_upper

# Function to plot odds ratios with inset
def plot_odds_ratios(course, odds_ratios, conf_intervals, zoom_in=False):
    reasons = ['Career', 'Interest', 'Degree', 'Other', 'Teaching', 'Know the Instructor']
    if course == 'ST1x' or course == 'UnixTx':
        reasons.remove('Know the Instructor')
    if course == 'FP101x':
        reasons.remove('Teaching')

    y_labels = [f'{reason}' for reason in reasons]
    y_pos = np.arange(len(y_labels))

    or_values = [odds_ratios[reason] for reason in reasons if reason in odds_ratios]
    ci_lowers = [conf_intervals[reason][0] for reason in reasons if reason in odds_ratios]
    ci_uppers = [conf_intervals[reason][1] for reason in reasons if reason in odds_ratios]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(or_values, y_pos[:len(or_values)], xerr=[np.subtract(or_values, ci_lowers), np.subtract(ci_uppers, or_values)], fmt='o', capsize=5, capthick=2)
    ax.axvline(x=1, color='grey', linestyle='--')
    ax.set_yticks(y_pos[:len(or_values)])
    ax.set_yticklabels(y_labels[:len(or_values)], fontsize=20)
    ax.tick_params(axis='x', labelsize=20)
    ax.set_xlabel('Odds Ratio (Male/Female)', fontsize=20)
    # ax.set_title(f'Odds Ratios for Reasons for Enrolment by Gender in {course}', fontsize=18)
    ax.invert_yaxis()

    # Add inset axes for zoomed-in plot
    if zoom_in and (course == 'FP101x' or course == 'EX101x' or course == 'All Courses'):
        inset_ax = inset_axes(ax, width="50%", height="50%", borderpad=3)
        reasons_zoom = ['Career', 'Interest', 'Degree', 'Other', 'Teaching']
        y_labels_zoom = [f'{reason}' for reason in reasons_zoom]
        y_pos_zoom = np.arange(len(y_labels_zoom))

        or_values_zoom = [odds_ratios[reason] for reason in reasons_zoom if reason in odds_ratios]
        ci_lowers_zoom = [conf_intervals[reason][0] for reason in reasons_zoom if reason in odds_ratios]
        ci_uppers_zoom = [conf_intervals[reason][1] for reason in reasons_zoom if reason in odds_ratios]

        inset_ax.errorbar(or_values_zoom, y_pos_zoom[:len(or_values_zoom)], xerr=[np.subtract(or_values_zoom, ci_lowers_zoom), np.subtract(ci_uppers_zoom, or_values_zoom)], fmt='o', capsize=5, capthick=2)
        inset_ax.axvline(x=1, color='grey', linestyle='--')
        inset_ax.set_yticks(y_pos_zoom[:len(or_values_zoom)])
        inset_ax.set_yticklabels(y_labels_zoom[:len(or_values_zoom)], fontsize=14)
        
        xticks = np.arange(0, max(ci_uppers_zoom)*1.1, 0.5)        
        inset_ax.set_xticks(xticks)
        inset_ax.set_xticklabels(xticks, fontsize=14)

        # inset_ax.tick_params(axis='x', labelsize=20)
        # inset_ax.set_xlim(0, 2.5)
        # inset_ax.set_xlabel('Odds Ratio', fontsize=16)
        inset_ax.invert_yaxis()
        inset_ax.spines['top'].set_linewidth(1.5)
        inset_ax.spines['right'].set_linewidth(1.5)
        inset_ax.spines['bottom'].set_linewidth(1.5)
        inset_ax.spines['left'].set_linewidth(1.5)

        # ax.set_title(f'Odds Ratios for Reasons for Enrolment by Gender in {course}', fontsize=18, x=0.45)


    plt.tight_layout()
    plt.savefig(f'{output_dir}/odds_ratios_{course}.png')
    plt.close()

# Prepare data for each course
for course, reasons in data.items():
    odds_ratios = {}
    conf_intervals = {}
    for reason, counts in reasons.items():
        men_count, women_count = counts
        or_value, ci_lower, ci_upper = calculate_odds_ratios(men_count, women_count, total_men[course], total_women[course])
        if or_value is not None:
            odds_ratios[reason] = or_value
            conf_intervals[reason] = (ci_lower, ci_upper)
    plot_odds_ratios(course, odds_ratios, conf_intervals, zoom_in=True)

# Combine data for all courses
combined_counts = {'Career': [0, 0], 'Interest': [0, 0], 'Degree': [0, 0], 'Other': [0, 0], 'Teaching': [0, 0], 'Know the Instructor': [0, 0]}
total_men_all = sum(total_men.values()) - data['EX101x']['Know the Instructor'][0] - data['FP101x']['Know the Instructor'][0]
total_women_all = sum(total_women.values()) - data['EX101x']['Know the Instructor'][1] - data['FP101x']['Know the Instructor'][1]

for course, reasons in data.items():
    for reason, counts in reasons.items():
        combined_counts[reason][0] += counts[0]
        combined_counts[reason][1] += counts[1]

combined_or = {}
combined_ci = {}

for reason, counts in combined_counts.items():
    men_count, women_count = counts
    or_value, ci_lower, ci_upper = calculate_odds_ratios(men_count, women_count, total_men_all, total_women_all)
    if or_value is not None:
        combined_or[reason] = or_value
        combined_ci[reason] = (ci_lower, ci_upper)

# Plot for all courses combined
plot_odds_ratios('All Courses', combined_or, combined_ci, zoom_in=True)

print("Graphs saved in the ./graphs directory.")
