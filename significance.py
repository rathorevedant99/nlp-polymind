import pandas as pd
from scipy.stats import wilcoxon, ttest_rel

# Load your data
data = pd.read_csv("./data/cumulative-performance-3-exp/run_data.csv")

# Prepare results list
results = []
alpha = 0.1  # Significance level

# Analyze each expert separately
for expert_id, expert_data in data.groupby('expert_id'):
    # Get paired before/after scores for this expert across all tasks/runs
    before_values = expert_data['before'].values
    after_values = expert_data['after'].values
    
    # Paired t-test
    t_stat, t_p_value = ttest_rel(before_values, after_values)
    
    # Wilcoxon signed-rank test
    w_stat, w_p_value = wilcoxon(before_values, after_values)
    
    # Determine significance
    t_significant = t_p_value < alpha
    w_significant = w_p_value < alpha
    
    # Store results
    results.append({
        'expert_id': expert_id,
        't_statistic': t_stat,
        't_p_value': t_p_value,
        't_significant': t_significant,
        'w_statistic': w_stat,
        'w_p_value': w_p_value,
        'w_significant': w_significant
    })

# Create results dataframe
results_df = pd.DataFrame(results)
print(results_df)