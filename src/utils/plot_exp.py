import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data = pd.read_csv("/workspace/nlp-polymind/outputs/2025-04-07/20-43-07/expert_run_performance.csv")

def plot_expert_run_performance(data: pd.DataFrame, save_path: str):
    # Calculate mean performance for each expert in each run
    mean_performance = data.groupby(['run', 'expert_id']).agg({
        'before': 'mean',
        'after': 'mean'
    }).reset_index()

    runs = sorted(mean_performance['run'].unique())
    experts = sorted(mean_performance['expert_id'].unique())

    x = np.arange(len(runs))
    width = 0.12  # Adjusted width
    n_experts = len(experts)
    total_width = width * 2 * n_experts  # Total width for each run's group of bars
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, expert in enumerate(experts):
        expert_data = mean_performance[mean_performance['expert_id'] == expert]
        # Calculate offset to center the groups
        group_offset = -total_width/2 + i * width * 2
        
        # Plot bars
        before_bars = ax.bar(x + group_offset, expert_data['before'], width, 
                            label=f'Expert {expert} Before', 
                            color=colors[i % len(colors)], alpha=0.6)
        after_bars = ax.bar(x + group_offset + width, expert_data['after'], width, 
                           label=f'Expert {expert} After', 
                           color=colors[i % len(colors)], alpha=0.9)

        # Add value labels
        # for j, (before, after) in enumerate(zip(expert_data['before'], expert_data['after'])):
        #     ax.text(x[j] + group_offset, before + 0.01, f'{before:.3f}', 
        #            ha='center', va='bottom', fontsize=8)
        #     ax.text(x[j] + group_offset + width, after + 0.01, f'{after:.3f}', 
        #            ha='center', va='bottom', fontsize=8)

    # Adjust plot settings
    ax.set_xticks(x)
    ax.set_xticklabels([f'Run {r}' for r in runs])
    ax.set_xlabel('Run')
    ax.set_ylabel('Average ROUGE-1 F Measure')
    ax.set_title('Average Expert Performance per Run')
    ax.legend(ncol=min(len(experts), 3), bbox_to_anchor=(0.5, -0.15), loc='upper center')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_expert_summary(data: pd.DataFrame, save_path: str):
    summary = data.groupby('expert_id').agg({
        'before': ['mean', 'std'],
        'after': ['mean', 'std']
    }).reset_index()

    # Flatten MultiIndex columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    x = np.arange(len(summary))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, summary['before_mean'], width, yerr=summary['before_std'], 
           capsize=5, label='Before', color='#3498db', alpha=0.7)
    ax.bar(x + width/2, summary['after_mean'], width, yerr=summary['after_std'], 
           capsize=5, label='After', color='#2ecc71', alpha=0.7)

    for i in x:
        ax.text(i - width/2, summary.loc[i, 'before_mean'] + 0.01, 
                f"{summary.loc[i, 'before_mean']:.3f}", ha='center', fontsize=8)
        ax.text(i + width/2, summary.loc[i, 'after_mean'] + 0.01, 
                f"{summary.loc[i, 'after_mean']:.3f}", ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Expert {i}' for i in summary['expert_id']])
    ax.set_xlabel('Expert')
    ax.set_ylabel('Average ROUGE-1 F Measure')
    ax.set_title('Overall Expert Performance')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()