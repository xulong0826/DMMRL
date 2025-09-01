import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments with specified GPU.")
    parser.add_argument('--gpu', type=int, default=1, help="GPU id to use for experiments")
    args = parser.parse_args()

    datasets = ["freesolv"]#"esol", "Lipophilicity",
    task_types = ["reg"]
    seeds = [1, 2025]
    norm = [0, 1]
    split_type = ["random"]#["random", "scaffold_balanced"]

    for run_number in range(1, 3):
        for dataset in datasets:
            for task in task_types:
                for seed in seeds:
                    for norm_value in norm:
                        for split in split_type:
                            cmd = (
                                f"python main.py --dataset {dataset} --task_type {task} "
                                f"--seed {seed} --gpu {args.gpu} --norm {norm_value} "
                                f"--split_type {split} --batch_size 64 --vib_norm {norm_value} "
                                f"--vib_hidden_dim 64 --vib_shared_dim 32 --vib_private_dim 32"
                            )
                            print(f"🚀 Running: {cmd}")
                            subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()