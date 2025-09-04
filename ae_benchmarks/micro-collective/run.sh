#!/bin/bash
# select_gpu_device wrapper script for HIP with extended compatibility
#export ROCR_VISIBLE_DEVICES="$OMPI_COMM_WORLD_LOCAL_RANK"
export HIP_VISIBLE_DEVICES=0

echo "FLUX_TASK_LOCAL_ID=${FLUX_TASK_LOCAL_ID}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"


exec "$@"