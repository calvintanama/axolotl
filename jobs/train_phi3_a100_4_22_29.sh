#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --job-name=ctanama-train
#SBATCH --constraint=LSDF
#SBATCH --ntasks=1 ### maybe 2 or 4
#SBATCH --mail-user="calvin.tanama@student.kit.edu"
#SBATCH --mail-type="ALL"
#SBATCH --output=/hkfs/work/workspace/scratch/cd7437-llmpruning/dump/slurm/train/%x_%j.out      ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=/hkfs/work/workspace/scratch/cd7437-llmpruning/dump/slurm/train/%x_%j.err       ### Slurm Error file, %x is job name, %j is job id
user_dir="/home/hk-project-test-p0023745/cd7437"
workspace_dir="/hkfs/work/workspace/scratch/cd7437-llmpruning"
cd "$user_dir"
### Load conda environment
source .bashrc
conda activate llmpruning_train2
### Change to the working directory
cd "$workspace_dir"
cd axolotl
### Run training
unset LD_LIBRARY_PATH
accelerate launch -m axolotl.cli.train ./config_run/phi3_pruned_extra_22_29_pretrain_a100_4.yaml