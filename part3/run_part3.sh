#!/bin/bash

# ============================================================
# Part 3: DPO (Direct Preference Optimization) Training
# ============================================================

echo "============================================================"
echo "Part 3: DPO Training"
echo "============================================================"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

# Configuration
MODEL_NAME="gpt2"
OUTPUT_DIR="./dpo_output"
MAX_SAMPLES=10000
EPOCHS=1
BATCH_SIZE=4
GRAD_ACCUM=2
LEARNING_RATE=5e-7
BETA=0.1  # DPO temperature parameter
MAX_LENGTH=512

echo ""
echo "Configuration:"
echo "  - Model: $MODEL_NAME"
echo "  - Max samples: $MAX_SAMPLES"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Gradient accumulation: $GRAD_ACCUM"
echo "  - Effective batch size: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Beta: $BETA"
echo ""

# Run DPO training
python dpo_trainer.py \
    --model_name $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --max_samples $MAX_SAMPLES \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --beta $BETA \
    --max_length $MAX_LENGTH \
    --logging_steps 50 \
    --save_steps 200

if [ $? -ne 0 ]; then
    echo "DPO training failed!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Part 3 COMPLETE!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  - Model: ${OUTPUT_DIR}/final/"
echo "  - Training curves: ${OUTPUT_DIR}/training_curves.png"
echo "  - History: ${OUTPUT_DIR}/training_history.json"
echo "  - Efficiency stats: ${OUTPUT_DIR}/efficiency_stats.json"