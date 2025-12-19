#! /bin/sh
#SBATCH --job-name=PATH
#SBATCH --output="PATH.out"
#SBATCH --error="PATH.err"
#SBATCH --partition=move,humanoid
#SBATCH --nodelist=move1,move2,move3,move4
#SBATCH --account=move
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 


# Set Warp cache directory to local tmp to avoid filesystem issues
# Warp uses HOME/.cache/warp by default, so create local HOME structure
ORIG_HOME=${HOME}
LOCAL_HOME="/tmp/${USER}_home_${SLURM_JOB_ID}"
mkdir -p ${LOCAL_HOME}/.cache/warp
export HOME=${LOCAL_HOME}
export WARP_CACHE_DIR=${LOCAL_HOME}/.cache/warp
# Save original HOME so Python can restore it after Warp initializes
export ORIG_HOME=${ORIG_HOME}

# YOUR PYTHON CMD HERE
