import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run multiple experiments with specified GPU.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU id to use for experiments")
    args = parser.parse_args()

    # 数据集分组
    group1 = ["bace", "bbbp", "clintox"]
    group2 = ["sider", "tox21"]
    task_types = ["class"]
    seeds = [1, 2025]
    norm = [0, 1]
    split_type = ["random"]

    for run_number in range(1, 3):
        for dataset in group1 + group2:
            # 根据分组设置特征维度
            if dataset in group1:
                vib_hidden_dim = 256
                vib_shared_dim = 128
                vib_private_dim = 128
            else:
                vib_hidden_dim = 64
                vib_shared_dim = 32
                vib_private_dim = 32
            for task in task_types:
                for seed in seeds:
                    for norm_value in norm:
                        for split in split_type:
                            cmd = (
                                f"python main.py --dataset {dataset} --task_type {task} "
                                f"--seed {seed} --gpu {args.gpu} --norm {norm_value} "
                                f"--split_type {split} --batch_size 64 --vib_norm {norm_value} "
                                f"--vib_hidden_dim {vib_hidden_dim} --vib_shared_dim {vib_shared_dim} --vib_private_dim {vib_private_dim}"
                            )
                            print(f"🚀 Running: {cmd}")
                            subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()