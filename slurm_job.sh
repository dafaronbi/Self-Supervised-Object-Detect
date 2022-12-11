#!/bin/bash
#SBATCH --job-name=obj_det
#SBATCH --output=object_detect.out
#SBATCH --partition=n1s16-v100-2
#SBATCH --mem=32GB
#SBATCH --account=ds_ga_1011-2022fa
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --requeue
singularity exec --nv --overlay /scratch/DL22FA/overlay_12-02.ext3:ro -B /scratch/DL22FA/unlabeled_112.sqsh:/unlabeled:image-src=/ -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ -B /scratch -B /scratch_tmp /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "source /ext3/env.sh; python FasterRCNN.py"
