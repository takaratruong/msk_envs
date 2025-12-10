#! /bin/sh
#SBATCH --job-name=PATH
#SBATCH --output="PATH.out"
#SBATCH --error="PATH.err"
#SBATCH --partition=move,humanoid
#SBATCH --nodelist=move3,move4
#SBATCH --account=move
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 


# YOUR PYTHON CMD HERE
