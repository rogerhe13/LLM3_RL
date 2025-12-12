#!/bin/bash

# ============================================================
# Part 2: Policy Optimization (PPO & GRPO)
# ============================================================

echo "============================================================"
echo "Part 2: Policy Optimization (PPO & GRPO)"
echo "============================================================"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# Configuration
BASE_MODEL="gpt2"
SFT_OUTPUT="./sft_output"
REWARD_MODEL_PATH="../part1/reward_model_output/best_model.pt"
NUM_STEPS=500
BATCH_SIZE=4          # Increased from 2
LEARNING_RATE=5e-7    # Lower learning rate for stability
MAX_SAMPLES=5000
MAX_NEW_TOKENS=64

# PPO specific
PPO_OUTPUT="./ppo_output"
CLIP_RATIO=0.2
KL_COEF=0.2           # Increased KL coefficient for stability
ENTROPY_COEF=0.01

# GRPO specific
GRPO_OUTPUT="./grpo_output"
GROUP_SIZE=4

echo ""
echo "Configuration:"
echo "  - Base Model: $BASE_MODEL"
echo "  - Reward Model: $REWARD_MODEL_PATH"
echo "  - Training steps: $NUM_STEPS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Max new tokens: $MAX_NEW_TOKENS"
echo ""

# ============================================================
# Step 0: SFT Training (Create initialization model)
# ============================================================
echo "============================================================"
echo "[Step 0/3] SFT Training (Creating initialization model)..."
echo "============================================================"

if [ -d "$SFT_OUTPUT" ]; then
    echo "SFT model already exists at $SFT_OUTPUT, skipping..."
else
    python sft_trainer.py \
        --model_name $BASE_MODEL \
        --output_dir $SFT_OUTPUT \
        --max_samples 10000 \
        --epochs 1 \
        --batch_size 8 \
        --learning_rate 2e-5

    if [ $? -ne 0 ]; then
        echo "SFT training failed!"
        exit 1
    fi
fi

echo ""

# ============================================================
# Step 1: Train with PPO (using SFT model)
# ============================================================
echo "============================================================"
echo "[Step 1/3] Training with PPO..."
echo "============================================================"

python ppo_trainer.py \
    --model_name $SFT_OUTPUT \
    --reward_model_path $REWARD_MODEL_PATH \
    --output_dir $PPO_OUTPUT \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_samples $MAX_SAMPLES \
    --clip_ratio $CLIP_RATIO \
    --kl_coef $KL_COEF \
    --entropy_coef $ENTROPY_COEF \
    --max_new_tokens $MAX_NEW_TOKENS

if [ $? -ne 0 ]; then
    echo "PPO training failed!"
    exit 1
fi

echo ""

# ============================================================
# Step 2: Train with GRPO (using SFT model)
# ============================================================
echo "============================================================"
echo "[Step 2/3] Training with GRPO..."
echo "============================================================"

python grpo_trainer.py \
    --model_name $SFT_OUTPUT \
    --reward_model_path $REWARD_MODEL_PATH \
    --output_dir $GRPO_OUTPUT \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --group_size $GROUP_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_samples $MAX_SAMPLES \
    --kl_coef $KL_COEF \
    --max_new_tokens $MAX_NEW_TOKENS

if [ $? -ne 0 ]; then
    echo "GRPO training failed!"
    exit 1
fi

echo ""

# ============================================================
# Step 3: Compare methods
# ============================================================
echo "============================================================"
echo "[Step 3/3] Comparing PPO vs GRPO..."
echo "============================================================"

python compare_methods.py \
    --ppo_output $PPO_OUTPUT \
    --grpo_output $GRPO_OUTPUT \
    --reward_model_path $REWARD_MODEL_PATH

echo ""
echo "============================================================"
echo "Part 2 COMPLETE!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  SFT:"
echo "    - Model: ${SFT_OUTPUT}/"
echo ""
echo "  PPO:"
echo "    - Model: ${PPO_OUTPUT}/final/"
echo "    - Training curves: ${PPO_OUTPUT}/training_curves.png"
echo "    - History: ${PPO_OUTPUT}/training_history.json"
echo ""
echo "  GRPO:"
echo "    - Model: ${GRPO_OUTPUT}/final/"
echo "    - Training curves: ${GRPO_OUTPUT}/training_curves.png"
echo "    - History: ${GRPO_OUTPUT}/training_history.json"
echo "    - Efficiency stats: ${GRPO_OUTPUT}/efficiency_stats.json"