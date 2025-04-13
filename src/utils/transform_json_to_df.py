import json
import pandas as pd
import os
root_path = os.getcwd()
target_folder = "outputs/2025-04-13/12-53-28"
with open(f'{root_path}/{target_folder}/run_data.json', 'r') as f:
    data = json.load(f)

rows = []

# Get the first available run ID dynamically
first_run_id = next(iter(data.keys()))
sample_task = data[first_run_id]['before']['0']
num_experts = len(sample_task)

for run_id, run_data in data.items():
    # For each task
    for task_id in range(10):  # Tasks 0-9
        task_id_str = str(task_id)
        
        # For each expert
        for expert_idx in range(num_experts):
            # Get this expert's value for this task
            before_value = run_data['before'][task_id_str][expert_idx]
            after_value = run_data['after'][task_id_str][expert_idx]
            
            # Add a row to our data
            rows.append({
                'run': int(run_id),
                'expert_id': expert_idx,
                'before': before_value,
                'after': after_value
            })

# Create DataFrame
df = pd.DataFrame(rows)

# Sort by run and expert_id
df = df.sort_values(['run', 'expert_id'])

# Save to CSV
df.to_csv(f'{root_path}/{target_folder}/run_data.csv', index=False)
print(f"\nData saved to run_data.csv at {root_path}/{target_folder}")