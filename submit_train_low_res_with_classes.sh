#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=gz_mnist
#SBATCH --ntasks=32
#SBATCH --time=5:00:00
#SBATCH --output=logs/gzmnist_low_res_x0_l2_with_classes/output_low_res%j
#SBATCH --error=logs/gzmnist_low_res_x0_l2_with_classes/error_low_res%j
#SBATCH --mail-user=coleman.krawczyk@port.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu.q
#SBATCH --nodelist=gpu01

module purge
module load system
module load anaconda3/2022.05
echo `module list`

source activate /mnt/lustre/shared_conda/envs/ckraw/gz-torch/
cd /users/ckraw/gz_diffusion/pytorch_diffusion_package
python train_gzmnist_low_res_with_classes.py
