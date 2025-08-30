import os
import subprocess

datasets = ["esol", "Lipophilicity"]
task_types = ["reg"]
seeds = [1, 42, 1024, 1234, 2025]
gpu = 0
norm = [0, 1]
split_type = ["random", "scaffold_balanced"]

for dataset in datasets:
    for task in task_types:
        for seed in seeds:
            for norm_value in norm:
                for split in split_type:
                    # Construct the command with the current parameters
                    cmd = f"python main.py --dataset {dataset} --task_type {task} --seed {seed} --gpu {gpu} --norm {norm_value} --split_type {split} --batch_size 64"
                    print(f"🚀 Running: {cmd}")
                    subprocess.run(cmd, shell=True)
