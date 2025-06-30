import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory if it doesn't exist
output_dir = './graphs'
os.makedirs(output_dir, exist_ok=True)

# Data from the table for each cluster and course
data = {
    'EX101x': {'Disengaging': [961, 297], 'Completing': [865, 322], 'Auditing': [2557, 916], 'Sampling': [25338, 10382]},
    'FP101x': {'Disengaging': [627, 23], 'Completing': [734, 29], 'Auditing': [1402, 79], 'Sampling': [11761, 1694]},
    'ST1x': {'Disengaging': [285, 116], 'Completing': [105, 29], 'Auditing': [428, 183], 'Sampling': [2976, 1338]},
    'UnixTx': {'Disengaging': [100, 21], 'Completing': [94, 13], 'Auditing': [436, 87], 'Sampling': [1342, 326]}
}

# Total number of learners for each course and gender
total_men = {'EX101x': 29721, 'FP101x': 14524, 'ST1x': 3794, 'UnixTx': 1972}
total_women = {'EX101x': 11917, 'FP101x': 1825, 'ST1x': 1666, 'UnixTx': 447}

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

# Calculate odds ratios and confidence intervals for each course
odds_ratios = {}
conf_intervals = {}

for course, clusters in data.items():
    odds_ratios[course] = {}
    conf_intervals[course] = {}
    for cluster, counts in clusters.items():
        men_count, women_count = counts
        or_value, ci_lower, ci_upper = calculate_odds_ratios(men_count, women_count, total_men[course], total_women[course])
        if or_value is not None:
            odds_ratios[course][cluster] = or_value
            conf_intervals[course][cluster] = (ci_lower, ci_upper)

# Function to plot odds ratios
def plot_odds_ratios(course, odds_ratios, conf_intervals):
    clusters = ['Auditing', 'Completing', 'Disengaging', 'Sampling']
    y_labels = [f'{cluster}' for cluster in clusters]
    y_pos = np.arange(len(y_labels))

    or_values = [odds_ratios[course][cluster] for cluster in clusters if cluster in odds_ratios[course]]
    ci_lowers = [conf_intervals[course][cluster][0] for cluster in clusters if cluster in odds_ratios[course]]
    ci_uppers = [conf_intervals[course][cluster][1] for cluster in clusters if cluster in odds_ratios[course]]

    plt.figure(figsize=(10, 6))
    plt.errorbar(or_values, y_pos[:len(or_values)], xerr=[np.subtract(or_values, ci_lowers), np.subtract(ci_uppers, or_values)], fmt='o', capsize=5, capthick=2)
    plt.axvline(x=1, color='grey', linestyle='--')
    plt.xlim(left=0, right=max(ci_uppers)*1.1)  # Adjust x-axis limits for better clarity
    plt.yticks(y_pos[:len(or_values)], y_labels[:len(or_values)], fontsize=20)  # Set y-label fontsize
    if max(ci_uppers) > 3:
        plt.xticks(np.arange(0, max(ci_uppers)*1.1, 1), fontsize=20)
    else:
        plt.xticks(np.arange(0, max(ci_uppers)*1.1, 0.5), fontsize=20)

    plt.xlabel('Odds Ratio (Male/Female)', fontsize=20)  # Set x-label fontsize
    # plt.title(f'Odds Ratios for Engagement Clusters by Gender in {course}', fontsize=18)  # Set title fontsize
    plt.gca().invert_yaxis()
    plt.tick_params(axis='x', labelsize=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/odds_ratios_{course}.png')
    plt.close()

# Plot for each course
for course in data.keys():
    plot_odds_ratios(course, odds_ratios, conf_intervals)

# Combine data for all courses
combined_counts = {'Disengaging': [0, 0], 'Completing': [0, 0], 'Auditing': [0, 0], 'Sampling': [0, 0]}
total_men_all = sum(total_men.values())
total_women_all = sum(total_women.values())

for course, clusters in data.items():
    for cluster, counts in clusters.items():
        combined_counts[cluster][0] += counts[0]
        combined_counts[cluster][1] += counts[1]

combined_or = {}
combined_ci = {}

for cluster, counts in combined_counts.items():
    men_count, women_count = counts
    or_value, ci_lower, ci_upper = calculate_odds_ratios(men_count, women_count, total_men_all, total_women_all)
    if or_value is not None:
        combined_or[cluster] = or_value
        combined_ci[cluster] = (ci_lower, ci_upper)

# Plot for all courses combined
clusters = ['Auditing', 'Completing', 'Disengaging', 'Sampling']
y_labels = [f'{cluster}' for cluster in clusters]
y_pos = np.arange(len(y_labels))

or_values = [combined_or[cluster] for cluster in clusters if cluster in combined_or]
ci_lowers = [combined_ci[cluster][0] for cluster in clusters if cluster in combined_or]
ci_uppers = [combined_ci[cluster][1] for cluster in clusters if cluster in combined_or]

plt.figure(figsize=(10, 6))
plt.errorbar(or_values, y_pos[:len(or_values)], xerr=[np.subtract(or_values, ci_lowers), np.subtract(ci_uppers, or_values)], fmt='o', capsize=5, capthick=2)
plt.axvline(x=1, color='grey', linestyle='--')
plt.xlim(left=0, right=max(ci_uppers)*1.1)  # Adjust x-axis limits for better clarity

plt.yticks(y_pos[:len(or_values)], y_labels[:len(or_values)], fontsize=20)  # Set y-label fontsize
plt.xlabel('Odds Ratio (Male/Female)', fontsize=20)  # Set x-label fontsize
# plt.title('Odds Ratios for Engagement Clusters by Gender Across All Courses', fontsize=20)  # Set title fontsize
plt.gca().invert_yaxis()
plt.xticks(np.arange(0, max(ci_uppers)*1.1, 0.5), fontsize=20)

plt.tick_params(axis='x', labelsize=20)

plt.tight_layout()
plt.savefig(f'{output_dir}/odds_ratios_all_courses.png')
plt.close()

print("Graphs saved in the ./graphs directory.")
