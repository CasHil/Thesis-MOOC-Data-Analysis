import matplotlib.pyplot as plt
import pandas as pd

# Data provided by the user
data = {
    "Course": ["EX101x", "FP101x", "ST1x", "UnixTx"],
    "Male": [41153, 46018, 9001, 2970],
    "Female": [16823, 6662, 3818, 595]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate percentages
df['Total'] = df['Male'] + df['Female']
df['Male %'] = df['Male'] / df['Total'] * 100
df['Female %'] = df['Female'] / df['Total'] * 100

# Plotting with specified colors and percentages
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = df.index

# Bar chart for male and female learners with specified colors
bar1 = plt.bar(index, df['Male'], bar_width, label='Male', color='#FFC20A')
bar2 = plt.bar(index + bar_width, df['Female'], bar_width, label='Female', color='#0C7BDC')

# Adding labels and title
plt.xlabel('Courses')
plt.ylabel('Number of Learners')
plt.title('Number of Male vs Female Learners in MOOCs')
plt.xticks(index + bar_width / 2, df['Course'])
plt.legend()

# Adding percentage labels on the bars
for i in index:
    plt.text(i, df['Male'][i] + 500, f"{df['Male %'][i]:.1f}%", ha='center', va='bottom', color='black')
    plt.text(i + bar_width, df['Female'][i] + 500, f"{df['Female %'][i]:.1f}%", ha='center', va='bottom', color='black')

# Display the plot
plt.tight_layout()
plt.savefig(".")