import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Set the style for better visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Read the data
df = pd.read_csv('data/4-Experts/run_data.csv')

# Calculate overall statistics
overall_wilcox = stats.wilcoxon(df['before'], df['after'])
overall_t = stats.ttest_rel(df['before'], df['after'])
overall_d = (df['after'].mean() - df['before'].mean()) / df['before'].std()

# Calculate additional statistics
n = len(df)
mean_diff = df['after'].mean() - df['before'].mean()
std_diff = np.std(df['after'] - df['before'])
se_diff = std_diff / np.sqrt(n)
ci_lower = mean_diff - 1.645 * se_diff  # 90% CI
ci_upper = mean_diff + 1.645 * se_diff

# Print results
print("\nOverall Improvement Analysis (α = 0.1):")
print("======================================")
print(f"Sample Size: {n}")
print(f"\nMean Before: {df['before'].mean():.3f} ± {df['before'].std():.3f}")
print(f"Mean After: {df['after'].mean():.3f} ± {df['after'].std():.3f}")
print(f"Mean Difference: {mean_diff:.3f} (90% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
print(f"Effect Size (Cohen's d): {overall_d:.3f}")
print(f"\nWilcoxon Test: p = {overall_wilcox[1]:.3f} {'*' if overall_wilcox[1] < 0.1 else ''}")
print(f"Paired t-test: p = {overall_t[1]:.3f} {'*' if overall_t[1] < 0.1 else ''}")

# Create a figure with multiple subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Box plot with individual points
sns.boxplot(x='variable', y='value', data=pd.melt(df, value_vars=['before', 'after']), ax=ax1)
sns.swarmplot(x='variable', y='value', data=pd.melt(df, value_vars=['before', 'after']), 
              color='red', alpha=0.5, size=4, ax=ax1)
ax1.set_title('Score Distribution')
ax1.set_xlabel('Time Point')
ax1.set_ylabel('Score')

# 2. Distribution of differences
differences = df['after'] - df['before']
sns.histplot(differences, kde=True, ax=ax2)
ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
ax2.axvline(x=mean_diff, color='green', linestyle='-', alpha=0.5)
ax2.fill_betweenx(ax2.get_ylim(), ci_lower, ci_upper, color='green', alpha=0.2)
ax2.set_title('Distribution of Score Differences')
ax2.set_xlabel('Difference (After - Before)')
ax2.set_ylabel('Count')

plt.tight_layout()
plt.savefig('detailed_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Print interpretation
print("\nInterpretation:")
print("===============")
print("1. Effect Size:")
if overall_d < 0.2:
    print("   - The effect size is negligible (d < 0.2)")
elif overall_d < 0.5:
    print("   - The effect size is small (0.2 ≤ d < 0.5)")
elif overall_d < 0.8:
    print("   - The effect size is medium (0.5 ≤ d < 0.8)")
else:
    print("   - The effect size is large (d ≥ 0.8)")

print("\n2. Statistical Significance:")
if overall_wilcox[1] < 0.1 or overall_t[1] < 0.1:
    print("   - There is statistically significant improvement")
else:
    print("   - There is no statistically significant improvement")
    print("   - This could be due to:")
    print("     a) Small effect size")
    print("     b) High variability in the data")
    print("     c) Insufficient sample size for the observed effect") 