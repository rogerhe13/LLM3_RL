"""
Part 2: Compare PPO vs GRPO
Generates comparison metrics and visualizations.
"""

import os
import json
import argparse
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from policy_model import PolicyModel, get_device

import sys
sys.path.append('../part1')
from reward_model import RewardModel


def load_training_history(output_dir: str) -> Dict:
    """Load training history from JSON."""
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'r') as f:
        return json.load(f)


def generate_samples(
    policy: PolicyModel,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    num_samples: int = 20
) -> List[Dict]:
    """Generate sample responses from a policy."""
    samples = []
    policy.eval()
    
    for i, prompt in enumerate(prompts[:num_samples]):
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = policy.generate(
                inputs['input_ids'],
                inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        samples.append({
            'prompt': prompt,
            'response': generated[len(prompt):].strip()
        })
    
    return samples


def evaluate_on_reward_model(
    policy: PolicyModel,
    reward_model: RewardModel,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    num_samples: int = 100
) -> List[float]:
    """Evaluate policy on reward model."""
    rewards = []
    policy.eval()
    
    for prompt in tqdm(prompts[:num_samples], desc="Evaluating"):
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = policy.generate(
                inputs['input_ids'],
                inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            attention_mask = (outputs != tokenizer.pad_token_id).long()
            reward = reward_model(outputs, attention_mask)
            rewards.append(reward.item())
    
    return rewards


def compare_methods(
    ppo_output_dir: str,
    grpo_output_dir: str,
    reward_model_path: str,
    output_dir: str = "./comparison_results"
):
    """Compare PPO and GRPO methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Important for decoder-only models
    
    # Load histories
    print("Loading training histories...")
    ppo_history = load_training_history(ppo_output_dir)
    grpo_history = load_training_history(grpo_output_dir)
    
    # Load models
    print("Loading models...")
    base_policy = PolicyModel("gpt2").to(device)
    
    ppo_policy = PolicyModel.load_pretrained(os.path.join(ppo_output_dir, "final")).to(device)
    grpo_policy = PolicyModel.load_pretrained(os.path.join(grpo_output_dir, "final")).to(device)
    
    reward_model = RewardModel.load_pretrained(reward_model_path, device=device)
    reward_model.eval()
    
    # Load test prompts
    print("Loading test prompts...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    test_prompts = []
    for item in dataset:
        text = item['chosen']
        parts = text.split("Assistant:")
        if len(parts) >= 2:
            prompt = "Assistant:".join(parts[:-1]).strip()
            if prompt:
                test_prompts.append(prompt + "\n\nAssistant:")
    test_prompts = test_prompts[:100]
    
    # Evaluate on reward model
    print("\nEvaluating models on reward model...")
    print("Evaluating base model...")
    base_rewards = evaluate_on_reward_model(base_policy, reward_model, tokenizer, test_prompts, device)
    print("Evaluating PPO model...")
    ppo_rewards = evaluate_on_reward_model(ppo_policy, reward_model, tokenizer, test_prompts, device)
    print("Evaluating GRPO model...")
    grpo_rewards = evaluate_on_reward_model(grpo_policy, reward_model, tokenizer, test_prompts, device)
    
    # Compute win rates
    print("\nComputing win rates...")
    ppo_vs_base = sum(1 for p, b in zip(ppo_rewards, base_rewards) if p > b) / len(base_rewards)
    grpo_vs_base = sum(1 for g, b in zip(grpo_rewards, base_rewards) if g > b) / len(base_rewards)
    ppo_vs_grpo = sum(1 for p, g in zip(ppo_rewards, grpo_rewards) if p > g) / len(ppo_rewards)
    
    # Generate samples
    print("\nGenerating sample responses...")
    base_samples = generate_samples(base_policy, tokenizer, test_prompts, device)
    ppo_samples = generate_samples(ppo_policy, tokenizer, test_prompts, device)
    grpo_samples = generate_samples(grpo_policy, tokenizer, test_prompts, device)
    
    # Load efficiency stats
    grpo_stats_path = os.path.join(grpo_output_dir, "efficiency_stats.json")
    if os.path.exists(grpo_stats_path):
        with open(grpo_stats_path, 'r') as f:
            grpo_efficiency = json.load(f)
    else:
        grpo_efficiency = {}
    
    # Create comparison report
    results = {
        'reward_statistics': {
            'base': {
                'mean': float(np.mean(base_rewards)),
                'std': float(np.std(base_rewards)),
                'min': float(np.min(base_rewards)),
                'max': float(np.max(base_rewards))
            },
            'ppo': {
                'mean': float(np.mean(ppo_rewards)),
                'std': float(np.std(ppo_rewards)),
                'min': float(np.min(ppo_rewards)),
                'max': float(np.max(ppo_rewards))
            },
            'grpo': {
                'mean': float(np.mean(grpo_rewards)),
                'std': float(np.std(grpo_rewards)),
                'min': float(np.min(grpo_rewards)),
                'max': float(np.max(grpo_rewards))
            }
        },
        'win_rates': {
            'ppo_vs_base': ppo_vs_base,
            'grpo_vs_base': grpo_vs_base,
            'ppo_vs_grpo': ppo_vs_grpo
        },
        'training_metrics': {
            'ppo': {
                'final_reward': float(np.mean(ppo_history['rewards'][-50:])),
                'final_kl': float(np.mean(ppo_history['kl_divergence'][-50:]))
            },
            'grpo': {
                'final_reward': float(np.mean(grpo_history['rewards'][-50:])),
                'final_kl': float(np.mean(grpo_history['kl_divergence'][-50:]))
            }
        },
        'efficiency': grpo_efficiency
    }
    
    # Save results
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save samples
    samples = {
        'base': base_samples,
        'ppo': ppo_samples,
        'grpo': grpo_samples
    }
    samples_path = os.path.join(output_dir, "generated_samples.json")
    with open(samples_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Create visualizations
    create_comparison_plots(ppo_history, grpo_history, base_rewards, ppo_rewards, grpo_rewards, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\nReward Model Scores (mean ± std):")
    print(f"  Base:  {np.mean(base_rewards):.4f} ± {np.std(base_rewards):.4f}")
    print(f"  PPO:   {np.mean(ppo_rewards):.4f} ± {np.std(ppo_rewards):.4f}")
    print(f"  GRPO:  {np.mean(grpo_rewards):.4f} ± {np.std(grpo_rewards):.4f}")
    
    print(f"\nWin Rates:")
    print(f"  PPO vs Base:  {ppo_vs_base:.1%}")
    print(f"  GRPO vs Base: {grpo_vs_base:.1%}")
    print(f"  PPO vs GRPO:  {ppo_vs_grpo:.1%}")
    
    print(f"\nFinal Training KL Divergence:")
    print(f"  PPO:  {results['training_metrics']['ppo']['final_kl']:.4f}")
    print(f"  GRPO: {results['training_metrics']['grpo']['final_kl']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def create_comparison_plots(
    ppo_history: Dict,
    grpo_history: Dict,
    base_rewards: List[float],
    ppo_rewards: List[float],
    grpo_rewards: List[float],
    output_dir: str
):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training rewards comparison
    axes[0, 0].plot(ppo_history['rewards'], alpha=0.7, label='PPO')
    axes[0, 0].plot(grpo_history['rewards'], alpha=0.7, label='GRPO')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards: PPO vs GRPO')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # KL divergence comparison
    axes[0, 1].plot(ppo_history['kl_divergence'], alpha=0.7, label='PPO')
    axes[0, 1].plot(grpo_history['kl_divergence'], alpha=0.7, label='GRPO')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence: PPO vs GRPO')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward distribution comparison
    axes[1, 0].boxplot(
        [base_rewards, ppo_rewards, grpo_rewards],
        labels=['Base', 'PPO', 'GRPO']
    )
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward Distribution on Test Set')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Policy loss comparison
    axes[1, 1].plot(ppo_history['policy_loss'], alpha=0.7, label='PPO')
    axes[1, 1].plot(grpo_history['policy_loss'], alpha=0.7, label='GRPO')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Policy Loss')
    axes[1, 1].set_title('Policy Loss: PPO vs GRPO')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_plots.png"), dpi=150)
    plt.close()
    
    print(f"Comparison plots saved to {output_dir}/comparison_plots.png")


def main():
    parser = argparse.ArgumentParser(description="Compare PPO vs GRPO")
    parser.add_argument("--ppo_output", type=str, default="./ppo_output")
    parser.add_argument("--grpo_output", type=str, default="./grpo_output")
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    
    args = parser.parse_args()
    
    compare_methods(
        args.ppo_output,
        args.grpo_output,
        args.reward_model_path,
        args.output_dir
    )


if __name__ == "__main__":
    main()