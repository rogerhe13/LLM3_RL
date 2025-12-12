# Assignment 3: Reinforcement Learning from Human Feedback

Student : Weihao He

## Overview

This project implements and compares three RLHF methods for aligning language models:

1. **PPO** (Proximal Policy Optimization) - Classic RL-based approach
2. **GRPO** (Group Relative Policy Optimization) - Group-based advantage estimation
3. **DPO** (Direct Preference Optimization) - Direct preference learning without RL

## Project Structure

```
.
├── part1/                      # Reward Model Training
│   ├── reward_model.py         # Reward model implementation
│   ├── train_reward_model.py   # Training script
│   └── reward_model_output/    # Trained model
│
├── part2/                      # PPO & GRPO Implementation
│   ├── policy_model.py         # Policy model wrapper
│   ├── ppo_trainer.py          # PPO training
│   ├── grpo_trainer.py         # GRPO training
│   ├── compare_methods.py      # Comparison script
│   └── run_part2.sh            # Run script
│
├── part3/                      # DPO Implementation
│   ├── dpo_trainer.py          # DPO training
│   └── run_part3.sh            # Run script
│
├── part4/                      # Analysis & Evaluation
│   ├── generate_samples.py     # Sample generation
│   ├── gpt4_judge.py           # GPT-4-as-judge evaluation
│   ├── analysis.py             # Analysis & plotting
│   ├── ANALYSIS.md             # Analysis report
│   └── run_part4.sh            # Run script
│
├── Dockerfile                  # Container definition
└── README.md                   # This file
```

## Requirements

### Hardware
- GPU with at least 16GB VRAM (tested on RTX 4090)
- 32GB+ RAM recommended

### Software
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA 11.8+ (for GPU support)

## Setup

### Using Docker (Recommended)

```bash
# Build the container
docker build -t rlhf-assignment .

# Run with GPU support
docker run --gpus all -it -v $(pwd):/workspace rlhf-assignment
```

### Manual Installation

```bash
# Create conda environment
conda create -n rlhf python=3.10
conda activate rlhf

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install transformers datasets numpy matplotlib tqdm openai
```

## Running the Assignment

### Part 1: Reward Model Training

```bash
cd part1
python train_reward_model.py \
    --model_name gpt2 \
    --max_samples 50000 \
    --epochs 3 \
    --batch_size 8
```

**Expected output**: Reward model saved to `part1/reward_model_output/best_model.pt`

### Part 2: PPO & GRPO Training

```bash
cd part2
bash run_part2.sh
```

Or run individually:

```bash
# PPO
python ppo_trainer.py \
    --model_name gpt2 \
    --reward_model_path ../part1/reward_model_output/best_model.pt \
    --num_steps 500 \
    --batch_size 2

# GRPO
python grpo_trainer.py \
    --model_name gpt2 \
    --reward_model_path ../part1/reward_model_output/best_model.pt \
    --num_steps 500 \
    --batch_size 2 \
    --group_size 4
```

**Expected output**:
- PPO model: `part2/ppo_output/final/`
- GRPO model: `part2/grpo_output/final/`
- Training curves: `*_output/training_curves.png`

### Part 3: DPO Training

```bash
cd part3
bash run_part3.sh
```

Or:

```bash
python dpo_trainer.py \
    --max_samples 10000 \
    --epochs 1 \
    --batch_size 4 \
    --beta 0.1
```

**Expected output**: DPO model saved to `part3/dpo_output/final/`

### Part 4: Analysis & Evaluation

```bash
cd part4

# Generate samples (requires GPU)
python generate_samples.py --num_samples 100

# Run GPT-4-as-judge (requires OpenAI API key)
export OPENAI_API_KEY=your_key_here
python gpt4_judge.py --num_samples 50

# Generate analysis plots
python analysis.py
```

**Expected output**:
- Samples: `part4/samples/`
- Figures: `part4/figures/`
- Analysis: `part4/ANALYSIS.md`

## Results Summary

| Method | Final Reward | Win Rate vs Base | Training Time |
|--------|-------------|------------------|---------------|
| Base   | -0.52       | -                | -             |
| PPO    | **0.63**    | **72%**          | 7.8 min       |
| GRPO   | 0.43        | 58%              | 6.2 min       |
| DPO    | 0.55        | 68%              | 12.5 min      |

## Key Findings

1. **PPO achieves highest reward** but requires careful tuning
2. **DPO provides best stability** with competitive performance
3. **GRPO underperforms** with small batch/group sizes
4. All methods improve significantly over the base model

See `part4/ANALYSIS.md` for detailed analysis.

## Compute Requirements

| Part | GPU Memory | Time (RTX 4090) |
|------|------------|-----------------|
| Part 1 (Reward Model) | ~8 GB | ~15 min |
| Part 2 (PPO) | ~10 GB | ~8 min |
| Part 2 (GRPO) | ~14 GB | ~6 min |
| Part 3 (DPO) | ~8 GB | ~15 min |
| Part 4 (Analysis) | ~6 GB | ~10 min |

**Total**: ~1 hour on RTX 4090

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce max_new_tokens
- Use gradient accumulation

### Training Instability
- Lower learning rate
- Increase KL coefficient
- Check for NaN/Inf in logs

### API Rate Limits
- Use gpt-3.5-turbo instead of gpt-4
- Add delays between API calls
- Reduce number of evaluation samples


## License

MIT License
