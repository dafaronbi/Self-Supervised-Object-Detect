#!/bin/bash
#SBATCH --job-name=vgg
#SBATCH --account=csci_ga_2572_2022fa_24
#SBATCH --partition=n1s8-v100-1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --output=./%j_%x.out
#SBATCH --error=./%j_%x.err
#SBATCH --time=2:00:00
#SBATCH -c 4
#SBATCH --requeue

singularity exec --nv --overlay /scratch/DL22FA/overlay_12-02.ext3:ro \
-B /scratch/DL22FA/unlabeled_112.sqsh:/unlabeled:image-src=/ \
-B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ \
-B /scratch/sy2199 -B /scratch_tmp/sy2199 \
/scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "
source /ext3/env.sh
conda activate project
python train.py
"