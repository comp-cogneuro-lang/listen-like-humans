#!/bin/bash

declare -A EXPERIMENTS

# Format: "config_path|run_name"
# Causal Models
EXPERIMENTS["baseline"]="experiments/en_words_ku_baseline.cfg|Baseline"
EXPERIMENTS["causal-cnn"]="experiments/en_words_ku_causal_cnn.cfg|causal-cnn"
EXPERIMENTS["causal-trans"]="experiments/en_words_ku_causal_trans.cfg|causal-trans"
EXPERIMENTS["causal-rcnn"]="experiments/en_words_ku_causal_rcnn.cfg|causal-rcnn"
EXPERIMENTS["causal-2lstm"]="experiments/en_words_ku_causal_2lstm.cfg|causal-2LSTM"
EXPERIMENTS["causal-ctrans"]="experiments/en_words_ku_causal_convtrans.cfg|causal-convtrans"

# Non-Causal Models
EXPERIMENTS["noncausal-2lstm"]="experiments/en_words_ku_2lstmbi.cfg|noncausal-2LSTM"
EXPERIMENTS["noncausal-convtrans"]="experiments/en_words_ku_convtrans.cfg|noncausal-convtrans"
EXPERIMENTS["noncausal-rcnn"]="experiments/en_words_ku_rcnn.cfg|noncausal-rcnn"
EXPERIMENTS["noncausal-cnn"]="experiments/en_words_ku_cnn.cfg|noncausal-cnn"
EXPERIMENTS["noncausal-trans"]="experiments/en_words_ku_trans.cfg|noncausal-trans"

# Get experiment key
KEY=$1

if [ -z "$KEY" ]; then
    echo "Usage: bash train.sh <experiment_key> [extra args for src/main.py]"
    echo "Available keys: ${!EXPERIMENTS[@]}"
    exit 1
fi

# Parse config
if [ -z "${EXPERIMENTS[$KEY]}" ]; then
    echo "Error: Unknown experiment key '$KEY'"
    echo "Available keys: ${!EXPERIMENTS[@]}"
    exit 1
fi

shift # remaining args are forwarded to src/main.py (e.g. --num_epochs 500)

IFS='|' read -r CONFIG_PATH RUN_NAME <<< "${EXPERIMENTS[$KEY]}"

echo "Running experiment '$KEY'"
echo "Config: $CONFIG_PATH"
echo "Name: $RUN_NAME"

# Run command
export USE_WANDB=${USE_WANDB:-1}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=src

python src/main.py \
    --config_path="$CONFIG_PATH" \
    --datapath=dataset \
    --name="$RUN_NAME" \
    --num_workers 2 \
    "$@"
