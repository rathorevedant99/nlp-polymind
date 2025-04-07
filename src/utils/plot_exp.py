import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = [
    {"run": 1, "expert_id": 0, "before": 0.33, "after": 0.51},
    {"run": 1, "expert_id": 1, "before": 0.38, "after": 0.41},
    {"run": 1, "expert_id": 2, "before": 0.31, "after": 0.35},
    {"run": 2, "expert_id": 0, "before": 0.46, "after": 0.44},
    {"run": 2, "expert_id": 1, "before": 0.41, "after": 0.43},
    {"run": 2, "expert_id": 2, "before": 0.38, "after": 0.34},
    {"run": 3, "expert_id": 0, "before": 0.32, "after": 0.45},
    {"run": 3, "expert_id": 1, "before": 0.42, "after": 0.42},
    {"run": 3, "expert_id": 2, "before": 0.30, "after": 0.35},
    {"run": 4, "expert_id": 0, "before": 0.39, "after": 0.45},
    {"run": 4, "expert_id": 1, "before": 0.38, "after": 0.42},
    {"run": 4, "expert_id": 2, "before": 0.41, "after": 0.34},
    {"run": 5, "expert_id": 0, "before": 0.35, "after": 0.449},
    {"run": 5, "expert_id": 1, "before": 0.41, "after": 0.42},
    {"run": 5, "expert_id": 2, "before": 0.37, "after": 0.35},
    {"run": 6, "expert_id": 0, "before": 0.35, "after": 0.44},
    {"run": 6, "expert_id": 1, "before": 0.42, "after": 0.42},
    {"run": 6, "expert_id": 2, "before": 0.43, "after": 0.34},
    {"run": 7, "expert_id": 0, "before": 0.35, "after": 0.45},
    {"run": 7, "expert_id": 1, "before": 0.30, "after": 0.42},
    {"run": 7, "expert_id": 2, "before": 0.36, "after": 0.35},
    {"run": 8, "expert_id": 0, "before": 0.39, "after": 0.45},
    {"run": 8, "expert_id": 1, "before": 0.30, "after": 0.42},
    {"run": 8, "expert_id": 2, "before": 0.37, "after": 0.35},
    {"run": 9, "expert_id": 0, "before": 0.34, "after": 0.45},
    {"run": 9, "expert_id": 1, "before": 0.39, "after": 0.42},
    {"run": 9, "expert_id": 2, "before": 0.45, "after": 0.36},
    {"run": 10, "expert_id": 0, "before": 0.4, "after": 0.448},
    {"run": 10, "expert_id": 1, "before": 0.365, "after": 0.422},
    {"run": 10, "expert_id": 2, "before": 0.456, "after": 0.3471},
]

def plot_expert_run_performance(data: pd.DataFrame, save_path: str):
    runs = sorted(data['run'].unique())
    experts = sorted(data['expert_id'].unique())

    x = np.arange(len(runs))
    width = 0.08
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(15, 8))

    for i, expert in enumerate(experts):
        d = data[data['expert_id'] == expert].sort_values('run')
        offset = i * width * 2.2

        ax.bar(x + offset, d['before'], width, label=f'Expert {expert} Before', color=colors[i % len(colors)], alpha=0.6)
        ax.bar(x + offset + width * 1.1, d['after'], width, label=f'Expert {expert} After', color=colors[i % len(colors)], alpha=0.9)

        for j, (b, a) in enumerate(zip(d['before'], d['after'])):
            if abs(b - a) > 0.01:
                ax.text(x[j] + offset, b + 0.01, f'{b:.2f}', ha='center', fontsize=8)
                ax.text(x[j] + offset + width * 1.1, a + 0.01, f'{a:.2f}', ha='center', fontsize=8)

    center_offset = len(experts) * width * 1.1
    ax.set_xticks(x + center_offset / 2)
    ax.set_xticklabels(runs)
    ax.set_xlabel('Run')
    ax.set_ylabel('ROUGE-1 F Measure')
    ax.set_title('Expert Performance per Run')
    ax.legend(ncol=min(len(experts), 3), bbox_to_anchor=(0.5, -0.1), loc='upper center')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
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
    ax.bar(x - width/2, summary['before_mean'], width, yerr=summary['before_std'], capsize=5, label='Before', color='#3498db', alpha=0.7)
    ax.bar(x + width/2, summary['after_mean'], width, yerr=summary['after_std'], capsize=5, label='After', color='#2ecc71', alpha=0.7)

    for i in x:
        ax.text(i - width/2, summary.loc[i, 'before_mean'] + 0.01, f"{summary.loc[i, 'before_mean']:.3f}", ha='center', fontsize=8)
        ax.text(i + width/2, summary.loc[i, 'after_mean'] + 0.01, f"{summary.loc[i, 'after_mean']:.3f}", ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f'Expert {i}' for i in summary['expert_id']])
    ax.set_xlabel('Expert')
    ax.set_ylabel('ROUGE-1 F Measure')
    ax.set_title('Mean Expert Performance')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()