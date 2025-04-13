import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

# Load your data into a DataFrame
data = pd.read_csv("./data/3-Experts/run_data.csv")

# Group the data by run and expert_id
grouped_data = data.groupby(['run', 'expert_id']).agg({'before': 'mean', 'after': 'mean'}).reset_index()
print(grouped_data.head())

# Prepare a list to store test results
results = []

# Define significance level
alpha = 0.2  # Using 0.2 as the significance level

# Loop over the grouped data
for (run, expert_id), group in grouped_data.groupby(['run', 'expert_id']):
    if len(group) > 1:  # Ensure there are at least two paired values
        before_values = group['before'].values
        after_values = group['after'].values
        
        # Apply paired t-test
        t_stat, t_p_value = ttest_rel(before_values, after_values)
        
        # Apply Wilcoxon signed-rank test
        w_stat, w_p_value = wilcoxon(before_values, after_values)
        
        # Check significance for t-test
        t_significant = t_p_value < alpha
        w_significant = w_p_value < alpha

        # Save the result for each run and expert_id
        results.append({
            'run': run,
            'expert_id': expert_id,
            't_statistic': t_stat,
            't_p_value': t_p_value,
            't_significant': t_significant,
            'w_statistic': w_stat,
            'w_p_value': w_p_value,
            'w_significant': w_significant
        })

# Convert results into a DataFrame for easy interpretation
results_df = pd.DataFrame(results)

# Show results
print(results_df)