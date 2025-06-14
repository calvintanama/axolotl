#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --job-name=residual-24-normal-ctanama
#SBATCH --constraint=LSDF
#SBATCH --ntasks=1 ### maybe 2 or 4
#SBATCH --mail-user="calvin.tanama@student.kit.edu"
#SBATCH --mail-type="ALL"
#SBATCH --output=/hkfs/work/workspace/scratch/cd7437-llmpruning_temp/dump/slurm/train/%x_%j.out      ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=/hkfs/work/workspace/scratch/cd7437-llmpruning_temp/dump/slurm/train/%x_%j.err       ### Slurm Error file, %x is job name, %j is job id
user_dir="/home/hk-project-test-p0023745/cd7437"
workspace_dir="/hkfs/work/workspace/scratch/cd7437-llmpruning_temp"
cd "$user_dir"
### Load conda environment
source .bashrc
conda activate llmpruning_train4
### Change to the working directory
cd "$workspace_dir"
cd axolotl_old/axolotl
### Run training
unset LD_LIBRARY_PATH
accelerate launch --config_file "$workspace_dir/axolotl_old/axolotl/accelerate/multi_gpu_config.yaml" -m axolotl.cli.train ./config_run/phi3_pruned_extra_pretrain_1_24_residual_24_a100_4_local.yaml