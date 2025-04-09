import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("/Users/vedantrathore/VSCode Working Folder/nlp-polymind/local/compile.csv")

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

def plot_expert_summary_every_n_runs(data: pd.DataFrame, n: int, save_path: str):
    # Get all unique runs and sort them
    all_runs = sorted(data['run'].unique())
    max_run = max(all_runs)
    
    # Calculate the number of summary points (every n runs)
    summary_points = list(range(n, max_run + 1, n))
    if max_run not in summary_points:
        summary_points.append(max_run)  # Always include the last run
    
    # Get unique experts
    experts = sorted(data['expert_id'].unique())
    
    # Create figure with dynamic size based on number of summary points
    fig_width = max(10, 2 * len(summary_points))
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    
    # Colors for different experts
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Width for bars
    width = 0.35
    n_experts = len(experts)
    
    # Calculate spacing between groups dynamically
    spacing = 0.5  # Base spacing
    
    # For each summary point (every n runs)
    for i, end_run in enumerate(summary_points):
        # Filter data up to this run
        subset_data = data[data['run'] <= end_run]
        
        # Calculate summary for this subset
        summary = subset_data.groupby('expert_id').agg({
            'before': ['mean'],
            'after': ['mean']
        }).reset_index()
        
        # Flatten MultiIndex columns
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Calculate x position for this group of bars
        group_offset = i * (4 * width + spacing)  # 4 bars per group (2 experts × 2 conditions)
        
        # Plot bars for each expert
        for j, expert in enumerate(experts):
            expert_data = summary[summary['expert_id'] == expert]
            
            # Calculate position within the group
            # For expert 0 and 1, place them next to each other
            if expert in [0, 1]:
                # Expert 0: left side, Expert 1: right side
                if expert == 0:
                    before_pos = group_offset - width
                    after_pos = group_offset - width/2
                else:  # expert == 1
                    before_pos = group_offset + width/2
                    after_pos = group_offset + width
            else:
                # For other experts, place them with spacing
                before_pos = group_offset + (j-1) * 2 * width
                after_pos = group_offset + (j-1) * 2 * width + width
            
            # Plot before bars
            before_bars = ax.bar(before_pos, 
                                expert_data['before_mean'].values[0], 
                                width, 
                                capsize=3,
                                label=f'Expert {expert} Before' if i == 0 else None,
                                color=colors[j % len(colors)], 
                                alpha=0.6)
            
            # Plot after bars
            after_bars = ax.bar(after_pos, 
                               expert_data['after_mean'].values[0], 
                               width, 
                               capsize=3,
                               label=f'Expert {expert} After' if i == 0 else None,
                               color=colors[j % len(colors)], 
                               alpha=0.9)
            
            # Add value labels
            ax.text(before_pos, 
                   expert_data['before_mean'].values[0] + 0.01, 
                   f"{expert_data['before_mean'].values[0]:.3f}", 
                   ha='center', va='top', fontsize=8)
            ax.text(after_pos, 
                   expert_data['after_mean'].values[0] + 0.01, 
                   f"{expert_data['after_mean'].values[0]:.3f}", 
                   ha='center', va='center', fontsize=8)
    
    # Set x-axis ticks and labels
    tick_positions = []
    tick_labels = []
    for i, end_run in enumerate(summary_points):
        center_pos = i * (4 * width + spacing) + width  # Center of the group
        tick_positions.append(center_pos)
        tick_labels.append(f'Run {end_run}')
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Set labels and title
    ax.set_xlabel('Cumulative Runs')
    ax.set_ylabel('Average ROUGE-1 F Measure')
    ax.set_title('Cumulative Performance over Successive Fine Tuning Runs')
    
    # Add legend
    ax.legend(ncol=min(len(experts), 3), bbox_to_anchor=(0.5, -0.15), loc='upper center')
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Fix the function call at the bottom
plot_expert_summary_every_n_runs(data, 5, "/Users/vedantrathore/VSCode Working Folder/nlp-polymind/local/compile_summary_every_5_runs.png")

# Comment out the incorrect function call
# plot_expert_summary_with_runs(data, "/Users/vedantrathore/VSCode Working Folder/nlp-polymind/local/compile_summary_with_runs.png")

# plot_expert_run_performance(data, "/Users/vedantrathore/VSCode Working Folder/nlp-polymind/local/compile_run_performance.png")
# plot_expert_summary(data, "/Users/vedantrathore/VSCode Working Folder/nlp-polymind/local/compile_summary.png")