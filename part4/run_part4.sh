#!/bin/bash

# ============================================================
# Part 4: Analysis and Evaluation
# ============================================================

echo "============================================================"
echo "Part 4: Analysis and Evaluation"
echo "============================================================"

OUTPUT_DIR="./output"
SAMPLES_DIR="./samples"
FIGURES_DIR="./figures"
REWARD_DIR="./reward_analysis"
FAILURE_DIR="./analysis_output"

mkdir -p $OUTPUT_DIR $SAMPLES_DIR $FIGURES_DIR $REWARD_DIR $FAILURE_DIR

# ============================================================
# Step 1: Generate samples from all models (requires GPU)
# ============================================================
echo ""
echo "[Step 1/5] Generating samples from all models..."
echo "============================================================"

python generate_samples.py \
    --base_model gpt2 \
    --ppo_model ../part2/ppo_output/final \
    --grpo_model ../part2/grpo_output/final \
    --dpo_model ../part3/dpo_output/final \
    --num_samples 100 \
    --output_dir $SAMPLES_DIR

# ============================================================
# Step 2: Compute reward distributions (requires GPU)
# ============================================================
echo ""
echo "[Step 2/5] Computing reward distributions..."
echo "============================================================"

python compute_rewards.py \
    --ppo_model ../part2/ppo_output/final \
    --grpo_model ../part2/grpo_output/final \
    --dpo_model ../part3/dpo_output/final \
    --reward_model_path ../part1/reward_model_output/best_model.pt \
    --num_prompts 100 \
    --output_dir $REWARD_DIR

# ============================================================
# Step 3: Failure mode analysis (requires GPU)
# ============================================================
echo ""
echo "[Step 3/5] Running failure mode analysis..."
echo "============================================================"

python failure_mode_analysis.py \
    --ppo_model ../part2/ppo_output/final \
    --grpo_model ../part2/grpo_output/final \
    --dpo_model ../part3/dpo_output/final \
    --output_dir $FAILURE_DIR

# ============================================================
# Step 4: GPT-4-as-Judge evaluation (requires API key)
# ============================================================
echo ""
echo "[Step 4/5] Running GPT-4-as-Judge evaluation..."
echo "============================================================"

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set. Skipping GPT-4 evaluation."
    echo "To run evaluation: export OPENAI_API_KEY=your_key"
else
    python gpt4_judge.py \
        --samples_file $SAMPLES_DIR/all_samples.json \
        --output_file $OUTPUT_DIR/evaluation_results.json \
        --judge_model gpt-3.5-turbo \
        --num_samples 50
fi

# ============================================================
# Step 5: Generate analysis and plots
# ============================================================
echo ""
echo "[Step 5/5] Generating analysis and plots..."
echo "============================================================"

python analysis.py \
    --ppo_history ../part2/ppo_output/training_history.json \
    --grpo_history ../part2/grpo_output/training_history.json \
    --dpo_history ../part3/dpo_output/training_history.json \
    --evaluation_results $OUTPUT_DIR/evaluation_results.json \
    --output_dir $FIGURES_DIR

echo ""
echo "============================================================"
echo "Part 4 COMPLETE!"
echo "============================================================"
echo ""
echo "Output files:"
echo "  Samples: $SAMPLES_DIR/"
echo "  Reward Analysis: $REWARD_DIR/"
echo "  Failure Mode: $FAILURE_DIR/"
echo "  Evaluation: $OUTPUT_DIR/evaluation_results.json"
echo "  Figures: $FIGURES_DIR/"
echo ""
echo "Reports:"
echo "  - $FAILURE_DIR/failure_mode_report.md"
echo "  - ANALYSIS.md"
echo ""
echo "Next: Review ANALYSIS.md and update with your findings!"
