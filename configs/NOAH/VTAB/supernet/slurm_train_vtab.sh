#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

PARTITION='dsta'
JOB_NAME=VTAB-SUPERNET
CONFIG=./experiments/NOAH/supernet/supernet-B_prompt.yaml
GPUS=1
CKPT=$1
WEIGHT_DECAY=0.0001

GPUS_PER_NODE=1
CPUS_PER_TASK=5
SRUN_ARGS=${SRUN_ARGS:-""}

mkdir -p logs
# SAFELY update PYTHONPATH
export PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH

# Provide fallback for RANDOM
RANDOM=${RANDOM:-42}

for LR in 0.0005; do 
    for DATASET in caltech101 dtd oxford_flowers102 dmlab; do 
        python supernet_train_prompt.py \
            --data-path=./data/vtab-1k/${DATASET} \
            --data-set=${DATASET} \
            --cfg=${CONFIG} \
            --resume=${CKPT} \
            --output_dir=./saves/${DATASET}_supernet_lr-${LR}_wd-${WEIGHT_DECAY} \
            --batch-size=64 \
            --lr=${LR} \
            --epochs=500 \
            --weight-decay=${WEIGHT_DECAY} \
            --no_aug \
            --direct_resize \
            --mixup=0 \
            --cutmix=0 \
            --smoothing=0 \
            --launcher="none" \
            2>&1 | tee logs/${currenttime}-${DATASET}-${LR}-vtab-supernet.log
        echo -e "\033[32m[ Please check log: \"logs/${currenttime}-${DATASET}-${LR}-vtab-supernet.log\" for details. ]\033[0m"
    done
done