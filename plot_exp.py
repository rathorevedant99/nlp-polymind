"""
 ____   ___    _   _  ___ _____   _   _ ____  _____   _____ _   _ ___ ____    ____ ___  ____  _____ 
|  _ \ / _ \  | \ | |/ _ \_   _| | | | / ___|| ____| |_   _| | | |_ _/ ___|  / ___/ _ \|  _ \| ____|
| | | | | | | |  \| | | | || |   | | | \___ \|  _|     | | | |_| || |\___ \ | |  | | | | | | |  _|  
| |_| | |_| | | |\  | |_| || |   | |_| |___) | |___    | | |  _  || | ___) || |__| |_| | |_| | |___ 
|____/ \___/  |_| \_|\___/ |_|    \___/|____/|_____|   |_| |_| |_|___|____/  \____\___/|____/|_____|
                                                                                                       
This is a genai script for interim plotting. DO NOT REUSE.
"""
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = {
    "Run": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Expert 0 Before": [0.33, 0.46, 0.32, 0.39, 0.35, 0.35, 0.35, 0.39, 0.34, 0.4],
    "Expert 1 Before": [0.38, 0.41, 0.42, 0.38, 0.41, 0.42, 0.30, 0.30, 0.39, 0.365],
    "Expert 2 Before": [0.31, 0.38, 0.30, 0.41, 0.37, 0.43, 0.36, 0.37, 0.45, 0.456],
    "Expert 0 After": [0.51, 0.44, 0.45, 0.45, 0.449, 0.44, 0.45, 0.45, 0.45, 0.448],
    "Expert 1 After": [0.41, 0.43, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.42, 0.422],
    "Expert 2 After": [0.35, 0.34, 0.35, 0.34, 0.35, 0.34, 0.35, 0.35, 0.36, 0.3471],
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# First subplot: Bar plot for each run
x = np.arange(len(df['Run']))
width = 0.15  # Width of the bars

# Plot bars for each expert before/after
ax1.bar(x - width*2.5, df['Expert 0 Before'], width, label='Expert 0 Before')
ax1.bar(x - width*1.5, df['Expert 0 After'], width, label='Expert 0 After')
ax1.bar(x - width*0.5, df['Expert 1 Before'], width, label='Expert 1 Before')
ax1.bar(x + width*0.5, df['Expert 1 After'], width, label='Expert 1 After')
ax1.bar(x + width*1.5, df['Expert 2 Before'], width, label='Expert 2 Before')
ax1.bar(x + width*2.5, df['Expert 2 After'], width, label='Expert 2 After')

ax1.set_xlabel('Run Number')
ax1.set_ylabel('Performance Score')
ax1.set_title('Performance by Run for Each Expert (Before vs After)')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Run'])
ax1.legend()

# Second subplot: Stacked bar plot with means and differences
means = {
    'Expert 0': {
        'Before': np.mean(df['Expert 0 Before']),
        'After': np.mean(df['Expert 0 After']),
        'Difference': np.mean(df['Expert 0 After']) - np.mean(df['Expert 0 Before']),
        'Before_std': np.std(df['Expert 0 Before']),
        'After_std': np.std(df['Expert 0 After'])
    },
    'Expert 1': {
        'Before': np.mean(df['Expert 1 Before']),
        'After': np.mean(df['Expert 1 After']),
        'Difference': np.mean(df['Expert 1 After']) - np.mean(df['Expert 1 Before']),
        'Before_std': np.std(df['Expert 1 Before']),
        'After_std': np.std(df['Expert 1 After'])
    },
    'Expert 2': {
        'Before': np.mean(df['Expert 2 Before']),
        'After': np.mean(df['Expert 2 After']),
        'Difference': np.mean(df['Expert 2 After']) - np.mean(df['Expert 2 Before']),
        'Before_std': np.std(df['Expert 2 Before']),
        'After_std': np.std(df['Expert 2 After'])
    }
}

experts = ['Expert 0', 'Expert 1', 'Expert 2']
x = np.arange(len(experts))
width = 0.35

# Create stacked bars with before and difference
before_means = [means[expert]['Before'] for expert in experts]
differences = [means[expert]['Difference'] for expert in experts]
before_std = [means[expert]['Before_std'] for expert in experts]
after_std = [means[expert]['After_std'] for expert in experts]

# Plot the base bars (Before values)
ax2.bar(x, before_means, width, label='Before', yerr=before_std, capsize=5)

# Plot the difference bars, with different colors based on positive/negative change
for i, (diff, before) in enumerate(zip(differences, before_means)):
    color = 'green' if diff > 0 else 'red'
    ax2.bar(x[i], diff, width, bottom=before, label='Improvement' if i == 0 and diff > 0 else 'Decline' if i == 0 and diff < 0 else '', color=color)

ax2.set_ylabel('Mean Performance Score')
ax2.set_title('Average Performance Summary\n(Base: Before Score, Stacked: Change in Performance)')
ax2.set_xticks(x)
ax2.set_xticklabels(experts)
ax2.legend()

# Add text annotations for the differences
for i, (diff, before) in enumerate(zip(differences, before_means)):
    if diff > 0:
        ax2.text(i, before + diff/2, f'+{diff:.3f}', ha='center', va='center')
    else:
        ax2.text(i, before + diff/2, f'{diff:.3f}', ha='center', va='center')

plt.tight_layout()
plt.savefig('expert_performance_comparison.png', bbox_inches='tight', dpi=300)
plt.close()
