#!/bin/bash

# ============================================================
# Part 1.2: Reward Model Training and Evaluation
# GPU Optimized Version (for NVIDIA GPUs like RTX 4000 Ada)
# ============================================================

echo "============================================================"
echo "Part 1.2: Reward Model Training and Evaluation (GPU)"
echo "============================================================"

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# Configuration - GPU Optimized
MODEL_NAME="gpt2"
MAX_LENGTH=512
MAX_SAMPLES=30000        # Training samples
EPOCHS=1                 # 1 epoch is usually enough for RLHF
BATCH_SIZE=8            # Larger batch size for GPU (16-32)
GRAD_ACCUM=2             # Effective batch size = BATCH_SIZE * GRAD_ACCUM = 32
LEARNING_RATE=1e-5
OUTPUT_DIR="./8_reward_model_output"
EVAL_OUTPUT_DIR="./8_evaluation_results"

# Use mixed precision training for faster training
USE_FP16="--fp16"

# Step 1: Train the Reward Model
echo ""
echo "[Step 1/2] Training Reward Model..."
echo "============================================================"
echo "Configuration:"
echo "  - Model: $MODEL_NAME"
echo "  - Max samples: $MAX_SAMPLES"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - Mixed precision: enabled"
echo "============================================================"

python train_reward_model.py \
    --model_name $MODEL_NAME \
    --max_length $MAX_LENGTH \
    --max_samples $MAX_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --output_dir $OUTPUT_DIR \
    --eval_steps 500 \
    --logging_steps 100 \
    $USE_FP16

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo "Training failed!"
    exit 1
fi

echo ""
echo "[Step 2/2] Evaluating Reward Model..."
echo "============================================================"

# Step 2: Evaluate and analyze errors
python evaluate_reward_model.py \
    --model_path "${OUTPUT_DIR}/best_model.pt" \
    --model_name $MODEL_NAME \
    --max_length $MAX_LENGTH \
    --batch_size 32 \
    --max_samples 5000 \
    --output_dir $EVAL_OUTPUT_DIR \
    --num_error_examples 25

echo ""
echo "============================================================"
echo "COMPLETE!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - Trained model: ${OUTPUT_DIR}/best_model.pt"
echo "  - Training curves: ${OUTPUT_DIR}/training_curves.png"
echo "  - Evaluation report: ${EVAL_OUTPUT_DIR}/evaluation_report.md"
echo "  - Error analysis: ${EVAL_OUTPUT_DIR}/evaluation_report.json"
echo "  - Reward distributions: ${EVAL_OUTPUT_DIR}/reward_distributions.png"
