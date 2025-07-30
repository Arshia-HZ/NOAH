# Set script to exit immediately if a command exits with a non-zero status.
set -e

# --- User-configurable variables ---
# Set the dataset you want to run the search on.
# The original script looped through a long list; here we run one at a time.
DATASET="oxford_flowers102" 

# Set the parameter limit. The paper uses 0.64M.
LIMITS="0.64"

# --- Fixed script configuration ---
# Path to the supernet configuration file.
CONFIG=./experiments/NOAH/supernet/supernet-B_prompt.yaml

# Create the output directory for this run.
OUTPUT_DIR="saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/search_limit-${LIMITS}"
mkdir -p ${OUTPUT_DIR}

# Path to the trained supernet checkpoint.
RESUME_PATH="saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/checkpoint.pth"

# Set the Python path to include the project's root directory.
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# --- Execute the evolutionary search ---
# The srun command is removed. We call python directly.
python evolution.py \
    --data-path=./data/vtab-1k/${DATASET} \
    --data-set=${DATASET} \
    --cfg=${CONFIG} \
    --output_dir=${OUTPUT_DIR} \
    --batch-size=64 \
    --resume=${RESUME_PATH} \
    --param-limits=${LIMITS} \
    --max-epochs=15 \
    --no_aug \
    --inception \
    --direct_resize \
    --mixup=0 \
    --cutmix=0 \
    --smoothing=0 \
    --launcher="none" # Changed from "slurm" to "none" for single-GPU execution.

echo "Search for ${DATASET} complete. Results are in ${OUTPUT_DIR}"