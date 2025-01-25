#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_4_a100
#SBATCH --job-name=ctanama-train
#SBATCH --constraint=LSDF
#SBATCH --ntasks=1 ### maybe 2 or 4
#SBATCH --mail-user="calvin.tanama@student.kit.edu"
#SBATCH --mail-type="ALL"
#SBATCH --output=/home/kit/anthropomatik/cd7437/dump/slurm/train/%x_%j.out      ### Slurm Output file, %x is job name, %j is job id
#SBATCH --error=/home/kit/anthropomatik/cd7437/dump/slurm/train/%x_%j.err       ### Slurm Error file, %x is job name, %j is job id
user_dir="/home/kit/anthropomatik/cd7437"
cd "$user_dir"
### Load conda environment
source .bashrc
conda activate llmpruning_gcc
### Change to the working directory
cd llmpruning/axolotl
### Run training
accelerate launch --num_processes 1 -m axolotl.cli.train ./config_run/phi3_pruned_extra_pretrain_a100_1.yaml