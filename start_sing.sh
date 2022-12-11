INSTANCE_NAME=${INSTANCE_NAME:-mycontainer}
singularity instance start --containall --no-home -B $HOME/.ssh --nv --overlay overlay_12-02.ext3 -B /scratch/DL22FA/unlabeled_112.sqsh:/unlabeled:image-src=/ -B /scratch/DL22FA/labeled.sqsh:/labeled:image-src=/ -B /scratch -B /scratch_tmp /scratch/DL22FA/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif ${INSTANCE_NAME}
