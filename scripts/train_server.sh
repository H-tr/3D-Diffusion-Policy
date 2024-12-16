#!/bin/bash
#SBATCH --job-name=dp3
#SBATCH --output=logs/kortex/pour.log
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianrunhu@gmail.com
#SBATCH --gres=gpu:1
#SBATCH --nodelist=crane2

bash scripts/train_policy.sh dp3 kortex_pour 0322 0 0

# Inference
# bash scripts/eval_policy.sh dp3 kortex_pour 0322 0 0