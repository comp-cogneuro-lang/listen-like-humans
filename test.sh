#!/bin/bash
#SBATCH --partition=general-gpu
#SBATCH --ntasks=20
#SBATCH --nodes=1
#SBATCH -C gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu[13-27]
#SBATCH --mem=100G
#SBATCH --output=%x_test_%j.out

# Load SLURM environment
# module load cuda/12.3
# source ~/.bashrc
# source ~/miniconda3/bin/activate bi
# nvidia-smi

# Define configurations
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
shift # Shift arguments to capture all subsequent arguments as epochs
EPOCHS="$@"

if [ -z "$KEY" ]; then
    echo "Usage: bash test.sh <experiment_key> <epoch1> [epoch2 ...]"
    echo "Available keys: ${!EXPERIMENTS[@]}"
    exit 1
fi

# Check if key is valid before proceeding
if [ -z "${EXPERIMENTS[$KEY]}" ]; then
    echo "Error: Unknown experiment key '$KEY'"
    echo "Available keys: ${!EXPERIMENTS[@]}"
    exit 1
fi

if [ -z "$EPOCHS" ]; then
    echo "Error: At least one epoch must be specified."
    
    # Try to find available checkpoints
    IFS='|' read -r CONFIG_PATH RUN_NAME <<< "${EXPERIMENTS[$KEY]}"
    CONFIG_DIR="${CONFIG_PATH%.cfg}"
    CHECKPOINT_DIR="$CONFIG_DIR/$RUN_NAME/pretraining"
    
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo "Available Checkpoints for '$KEY':"
        # List files, extract numbers, sort numerically
        ls "$CHECKPOINT_DIR"/model_state_*.pth 2>/dev/null | grep -o 'model_state_[0-9]\+\.pth' | grep -o '[0-9]\+' | sort -n | tr '\n' ' '
        echo "" # New line
    else
        ALT_CHECKPOINT_DIR="experiments/${CONFIG_PATH##*/}" # Strips path, keeps filename e.g. en_words_ku_baseline.cfg
        ALT_CHECKPOINT_DIR="${ALT_CHECKPOINT_DIR%.cfg}/$RUN_NAME/pretraining"
        
        if [ -d "$ALT_CHECKPOINT_DIR" ]; then
             echo "Available Checkpoints for '$KEY' (found at alternate path):"
             ls "$ALT_CHECKPOINT_DIR"/model_state_*.pth 2>/dev/null | grep -o 'model_state_[0-9]\+\.pth' | grep -o '[0-9]\+' | sort -n | tr '\n' ' '
             echo ""
        else
            echo "No checkpoint directory found at: $CHECKPOINT_DIR"
            echo "Also checked: $ALT_CHECKPOINT_DIR"
        fi
    fi
    exit 1
fi

IFS='|' read -r CONFIG_PATH RUN_NAME <<< "${EXPERIMENTS[$KEY]}"

echo "Running test for experiment '$KEY' on epochs: $EPOCHS"
echo "Config: $CONFIG_PATH"
echo "Name: $RUN_NAME"

ADD_ARGS=""
if [[ "$KEY" == *"trans"* ]]; then
    ADD_ARGS="--causal_transformer"
fi

python analysis/_comp_competition_batch.py \
    --config_path "$CONFIG_PATH" \
    --name "$RUN_NAME" \
    --epochs $EPOCHS \
    $ADD_ARGS
