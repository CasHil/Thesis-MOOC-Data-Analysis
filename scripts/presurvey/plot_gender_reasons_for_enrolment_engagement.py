import numpy as np
import scipy.stats as stats

# Data from the table for men
contingency_table_men = np.array([
    [609, 669, 946, 1641],  # Men - Career
    [41, 44, 84, 126],      # Men - Degree
    [279, 386, 532, 811],   # Men - Interest
    [21, 36, 28, 29],       # Men - Know the Instructor
    [15, 10, 12, 31],       # Men - Other
    [4, 4, 9, 10]           # Men - Teaching
])

# Data from the table for women
contingency_table_women = np.array([
    [193, 119, 397, 818],   # Women - Career
    [12, 11, 26, 46],       # Women - Degree
    [38, 34, 74, 198],      # Women - Interest
    [0, 1, 0, 3],           # Women - Know the Instructor
    [3, 1, 6, 16],          # Women - Other
    [2, 0, 3, 5]            # Women - Teaching
])

# Function to calculate Cohen's omega and Cramér's V


def calculate_effect_sizes(contingency_table):
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    n = np.sum(contingency_table)  # Total sample size
    # Number of categories in the variable with the fewest categories
    k = min(contingency_table.shape)
    cramers_v = np.sqrt(chi2 / (n * (k - 1)))
    cohens_omega = np.sqrt((chi2 / n) / (k - 1))
    return chi2, p, dof, cramers_v, cohens_omega


# Calculate for men
chi2_men, p_men, dof_men, cramers_v_men, cohens_omega_men = calculate_effect_sizes(
    contingency_table_men)
print(f"Men - Chi-Square: {chi2_men}")
print(f"Men - p-value: {p_men}")
print(f"Men - Degrees of Freedom: {dof_men}")
print(f"Men - Cramér's V: {cramers_v_men}")
print(f"Men - Cohen's Omega: {cohens_omega_men}")
if cohens_omega_men < 0.1:
    print("Men - Weak association")
elif cohens_omega_men < 0.3:
    print("Men - Moderate association")
else:
    print("Men - Strong association")

# Calculate for women
chi2_women, p_women, dof_women, cramers_v_women, cohens_omega_women = calculate_effect_sizes(
    contingency_table_women)
print(f"Women - Chi-Square: {chi2_women}")
print(f"Women - p-value: {p_women}")
print(f"Women - Degrees of Freedom: {dof_women}")
print(f"Women - Cramér's V: {cramers_v_women}")
print(f"Women - Cohen's Omega: {cohens_omega_women}")
if cohens_omega_women < 0.1:
    print("Women - Weak association")
elif cohens_omega_women < 0.3:
    print("Women - Moderate association")
else:
    print("Women - Strong association")
