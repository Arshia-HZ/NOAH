# Exit immediately if a command exits with a non-zero status.
set -e

# --- User-configurable variables ---
# Set the dataset you want to retrain on.
DATASET="oxford_flowers102"

# IMPORTANT: Set the path to your trained SUPERNET checkpoint.
# The retrain process loads the shared weights from the supernet
# and fine-tunes the specific subnet architecture.
CKPT="saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/checkpoint.pth"

# Set training hyperparameters.
LR=0.001
WEIGHT_DECAY=0.0001

# --- Fixed script configuration ---
# Path to the SUBNET configuration file, which defines the architecture found during search.
CONFIG="experiments/NOAH/subnet/VTAB/ViT-B_prompt_${DATASET}.yaml"

# Create the output directory for this run.
OUTPUT_DIR="saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/retrain_${LR}_wd-${WEIGHT_DECAY}"
mkdir -p ${OUTPUT_DIR}

# Set the Python path to include the project's root directory.
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# --- Execute the retraining ---
# The srun command is removed. We call python directly.
python supernet_train_prompt.py \
    --data-path=./data/vtab-1k/${DATASET} \
    --data-set=${DATASET} \
    --cfg=${CONFIG} \
    --resume=${CKPT} \
    --output_dir=${OUTPUT_DIR} \
    --batch-size=64 \
    --mode=retrain \
    --epochs=100 \
    --lr=${LR} \
    --weight-decay=${WEIGHT_DECAY} \
    --no_aug \
    --direct_resize \
    --mixup=0 \
    --cutmix=0 \
    --smoothing=0 \
    --launcher="none" # Changed from "slurm" to "none" for single-GPU execution.

echo "Retraining for ${DATASET} complete. Final model saved in ${OUTPUT_DIR}"