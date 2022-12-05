#!/bin/bash
#SBATCH --job-name=vgg
#SBATCH --account=csci_ga_2572_2022fa_24
#SBATCH --partition=n1s16-v100-2
#SBATCH --gres=gpu:1
#SBATCH --output=vgg.out
#SBATCH --error=vgg.err
#SBATCH --time=0:05:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --requeue

singularity exec --nv --overlay /scratch/DL22FA/overlay_11-16.ext3:ro \
-B /scratch/DL22FA/unlabeled_112.sqsh:/unlabeled:image-src=/ \
-B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
-B /scratch -B /scratch_tmp \
/scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif 
/bin/bash -c "
source /ext3/env.sh
conda activate project
python train.py
”