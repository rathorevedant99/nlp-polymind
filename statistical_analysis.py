import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, norm
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for better visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 8]

# Function to perform randomization test
def randomization_test(before, after, n_permutations=1000):
    """
    Perform a randomization test on paired data
    
    Parameters:
    -----------
    before : array-like
        Before scores
    after : array-like
        After scores
    n_permutations : int
        Number of random permutations to perform
        
    Returns:
    --------
    p_value : float
        The p-value from the randomization test
    """
    # Calculate the observed test statistic (mean difference)
    observed_diff = np.mean(after - before)
    
    # Initialize counter for more extreme differences
    count = 0
    
    # Perform randomizations
    for _ in range(n_permutations):
        # Randomly decide whether to swap each pair
        swaps = np.random.choice([-1, 1], size=len(before))
        permuted_diff = np.mean(swaps * (after - before))
        
        # Count if this difference is as extreme as observed
        if abs(permuted_diff) >= abs(observed_diff):
            count += 1
    
    # Calculate p-value
    p_value = (count + 1) / (n_permutations + 1)
    
    return round(p_value, 3)

# Read the data
base_path = "./data/30-runs/"
file = "2_experts.csv"

df = pd.read_csv(base_path + file)

# Calculate overall statistics
before_mean = df['before'].mean()
after_mean = df['after'].mean()
before_std = df['before'].std()
after_std = df['after'].std()
mean_diff = after_mean - before_mean
effect_size = mean_diff / np.sqrt((before_std**2 + after_std**2) / 2)

# Perform statistical tests on overall data
w_stat, w_p_value = wilcoxon(df['before'], df['after'])
t_stat, t_p_value = ttest_rel(df['before'], df['after'])
r_p_value = randomization_test(df['before'].values, df['after'].values)

# Calculate confidence interval for the mean difference
diff_std = df['after'] - df['before']
diff_std_err = diff_std.std() / np.sqrt(len(diff_std))
ci_lower = mean_diff - 1.645 * diff_std_err  # 90% CI
ci_upper = mean_diff + 1.645 * diff_std_err  # 90% CI

# Print overall results
print("\n=== OVERALL IMPROVEMENT ANALYSIS ===")
print(f"Sample Size: {len(df)}")
print(f"Mean Before: {before_mean:.5f} ± {before_std:.5f}")
print(f"Mean After: {after_mean:.5f} ± {after_std:.5f}")
print(f"Mean Difference: {mean_diff:.5f} [90% CI: {ci_lower:.5f}, {ci_upper:.5f}]")
print(f"Effect Size (Cohen's d): {effect_size:.5f}")
print(f"Wilcoxon Test: p = {w_p_value:.5f} {'*' if w_p_value < 0.05 else ''}")
print(f"Paired t-test: p = {t_p_value:.5f} {'*' if t_p_value < 0.05 else ''}")
print(f"Randomization Test: p = {r_p_value:.5f} {'*' if r_p_value < 0.05 else ''}")
print("* indicates statistical significance at α = 0.05")

# Analyze each expert separately
print("\n=== INDIVIDUAL EXPERT ANALYSIS ===")
expert_results = []

for expert_id, expert_data in df.groupby('expert_id'):
    # Calculate statistics for this expert
    exp_before_mean = expert_data['before'].mean()
    exp_after_mean = expert_data['after'].mean()
    exp_before_std = expert_data['before'].std()
    exp_after_std = expert_data['after'].std()
    exp_mean_diff = exp_after_mean - exp_before_mean
    exp_effect_size = exp_mean_diff / np.sqrt((exp_before_std**2 + exp_after_std**2) / 2)
    
    # Perform statistical tests
    exp_w_stat, exp_w_p_value = wilcoxon(expert_data['before'], expert_data['after'])
    exp_t_stat, exp_t_p_value = ttest_rel(expert_data['before'], expert_data['after'])
    exp_r_p_value = randomization_test(expert_data['before'].values, expert_data['after'].values)
    
    # Calculate confidence interval
    exp_diff_std = expert_data['after'] - expert_data['before']
    exp_diff_std_err = exp_diff_std.std() / np.sqrt(len(exp_diff_std))
    exp_ci_lower = exp_mean_diff - 1.645 * exp_diff_std_err
    exp_ci_upper = exp_mean_diff + 1.645 * exp_diff_std_err
    
    # Store results
    expert_results.append({
        'expert_id': expert_id,
        'sample_size': len(expert_data),
        'before_mean': exp_before_mean,
        'before_std': exp_before_std,
        'after_mean': exp_after_mean,
        'after_std': exp_after_std,
        'mean_diff': exp_mean_diff,
        'ci_lower': exp_ci_lower,
        'ci_upper': exp_ci_upper,
        'effect_size': exp_effect_size,
        'wilcoxon_p': exp_w_p_value,
        'ttest_p': exp_t_p_value,
        'randomization_p': exp_r_p_value,
        'significant': exp_w_p_value < 0.05 or exp_t_p_value < 0.05 or exp_r_p_value < 0.05
    })
    
    # Print results for this expert
    print(f"\nExpert {expert_id}:")
    print(f"  Sample Size: {len(expert_data)}")
    print(f"  Mean Before: {exp_before_mean:.5f} ± {exp_before_std:.5f}")
    print(f"  Mean After: {exp_after_mean:.5f} ± {exp_after_std:.5f}")
    print(f"  Mean Difference: {exp_mean_diff:.5f} [90% CI: {exp_ci_lower:.5f}, {exp_ci_upper:.5f}]")
    print(f"  Effect Size (Cohen's d): {exp_effect_size:.5f}")
    print(f"  Wilcoxon Test: p = {exp_w_p_value:.5f} {'*' if exp_w_p_value < 0.05 else ''}")
    print(f"  Paired t-test: p = {exp_t_p_value:.5f} {'*' if exp_t_p_value < 0.05 else ''}")
    print(f"  Randomization Test: p = {exp_r_p_value:.5f} {'*' if exp_r_p_value < 0.05 else ''}")
    if exp_w_p_value < 0.05 or exp_t_p_value < 0.05 or exp_r_p_value < 0.05:
        print("  ** Statistically significant improvement **")

# Convert to DataFrame for easier analysis
expert_df = pd.DataFrame(expert_results)

# Create visualizations
plt.figure(figsize=(15, 10))

# 1. Box plot with individual points for each expert
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='expert_id', y='before', color='lightblue', width=0.3)
sns.boxplot(data=df, x='expert_id', y='after', color='lightgreen', width=0.3)
sns.stripplot(data=df, x='expert_id', y='before', color='blue', alpha=0.3, size=3)
sns.stripplot(data=df, x='expert_id', y='after', color='green', alpha=0.3, size=3)
plt.title('Score Distribution by Expert')
plt.xlabel('Expert ID')
plt.ylabel('Score')

# 2. Effect size by expert
plt.subplot(2, 2, 2)
sns.barplot(data=expert_df, x='expert_id', y='effect_size', palette='viridis')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Effect Size by Expert')
plt.xlabel('Expert ID')
plt.ylabel("Cohen's d")

# 3. P-values by expert
plt.subplot(2, 2, 3)
expert_df_melted = pd.melt(expert_df, id_vars=['expert_id'], 
                           value_vars=['wilcoxon_p', 'ttest_p', 'randomization_p'],
                           var_name='test', value_name='p_value')
sns.barplot(data=expert_df_melted, x='expert_id', y='p_value', hue='test')
plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='α = 0.05')
plt.title('P-values by Expert and Test')
plt.xlabel('Expert ID')
plt.ylabel('P-value')
plt.legend()

# 4. Mean difference with confidence intervals
plt.subplot(2, 2, 4)
sns.barplot(data=expert_df, x='expert_id', y='mean_diff', palette='viridis')
plt.errorbar(x=expert_df['expert_id'], y=expert_df['mean_diff'], 
             yerr=[expert_df['mean_diff'] - expert_df['ci_lower'], expert_df['ci_upper'] - expert_df['mean_diff']],
             fmt='none', color='black', capsize=5)
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.title('Mean Difference with 90% CI by Expert')
plt.xlabel('Expert ID')
plt.ylabel('Mean Difference (After - Before)')

plt.tight_layout()
plt.savefig(base_path + f'{file.split(".")[0]}_expert_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a summary table
summary_df = expert_df[['expert_id', 'sample_size', 'before_mean', 'after_mean', 'mean_diff', 'effect_size', 'wilcoxon_p', 'ttest_p', 'randomization_p', 'significant']]
summary_df.columns = ['Expert ID', 'Sample Size', 'Before Mean', 'After Mean', 'Mean Diff', 'Effect Size', 'Wilcoxon p', 't-test p', 'Randomization p', 'Significant']
print("\n=== SUMMARY TABLE ===")
print(summary_df.to_string(index=False))

# Calculate percentage of experts with significant improvement
significant_count = expert_df['significant'].sum()
significant_percentage = (significant_count / len(expert_df)) * 100
print(f"\nPercentage of experts with significant improvement: {significant_percentage:.1f}% ({significant_count}/{len(expert_df)})")

# Save summary to txt
with open(f'{base_path}expert_analysis_{file.split(".")[0]}.txt', 'w') as f:
    f.write("=== SUMMARY TABLE ===")
    f.write(summary_df.to_string(index=False))
    f.write(f"\nPercentage of experts with significant improvement: {significant_percentage:.1f}% ({significant_count}/{len(expert_df)})\n")
    f.write("\n=== OVERALL IMPROVEMENT ANALYSIS ===")
    f.write(f"\nSample Size: {len(df)}\n")
    f.write(f"\nMean Before: {before_mean:.5f} ± {before_std:.5f}\n")
    f.write(f"\nMean After: {after_mean:.5f} ± {after_std:.5f}\n")
    f.write(f"\nMean Difference: {mean_diff:.5f} [90% CI: {ci_lower:.5f}, {ci_upper:.5f}]\n")
    f.write(f"\nEffect Size (Cohen's d): {effect_size:.5f}\n")
    f.write(f"\nWilcoxon Test: p = {w_p_value:.5f} {'*' if w_p_value < 0.05 else ''}\n")
    f.write(f"\nPaired t-test: p = {t_p_value:.5f} {'*' if t_p_value < 0.05 else ''}\n")
    f.write(f"\nRandomization Test: p = {r_p_value:.5f} {'*' if r_p_value < 0.05 else ''}\n")
    if w_p_value < 0.05 or t_p_value < 0.05 or r_p_value < 0.05:
        f.write("  ** Statistically significant improvement **\n\n")
    f.write("\n=== INDIVIDUAL EXPERT ANALYSIS ===")
    for _, row in expert_df.iterrows():
        f.write(f"\nExpert {row['expert_id']}:\n")
        f.write(f"  Sample Size: {row['sample_size']}\n")
        f.write(f"  Mean Before: {row['before_mean']:.5f}\n")
        f.write(f"  Mean After: {row['after_mean']:.5f}\n")
        f.write(f"  Mean Difference: {row['mean_diff']:.5f}\n")
        f.write(f"  Effect Size (Cohen's d): {row['effect_size']:.5f}\n")
        f.write(f"  Wilcoxon Test: p = {row['wilcoxon_p']:.5f} {'*' if row['wilcoxon_p'] < 0.05 else ''}\n")
        f.write(f"  Paired t-test: p = {row['ttest_p']:.5f} {'*' if row['ttest_p'] < 0.05 else ''}\n")
        f.write(f"  Randomization Test: p = {row['randomization_p']:.5f} {'*' if row['randomization_p'] < 0.05 else ''}\n")
        if row['wilcoxon_p'] < 0.05 or row['ttest_p'] < 0.05 or row['randomization_p'] < 0.05:
            f.write("  ** Statistically significant improvement **\n")
    f.write("\n")

print(f"\nSummary saved to '{base_path}expert_analysis_{file.split('.')[0]}.txt'")