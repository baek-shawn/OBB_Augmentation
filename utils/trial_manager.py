import os

def get_new_trial_path(root_output="output"):
    os.makedirs(root_output, exist_ok=True)
    trials = [d for d in os.listdir(root_output) if d.startswith("trial_") and os.path.isdir(os.path.join(root_output, d))]
    
    trial_nums = []
    for trial in trials:
        try:
            num = int(trial.split("_")[1])
            trial_nums.append(num)
        except (IndexError, ValueError):
            continue
    
    next_trial_num = max(trial_nums) + 1 if trial_nums else 1
    new_trial_name = f"trial_{next_trial_num}"
    new_trial_path = os.path.join(root_output, new_trial_name)
    
    os.makedirs(new_trial_path, exist_ok=True)
    return new_trial_path
