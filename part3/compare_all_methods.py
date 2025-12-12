"""
Part 3: Compare DPO with PPO and GRPO
Generate comparison metrics and sample outputs.
"""

import os
import json
import argparse
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path: str, device: torch.device):
    """Load a trained model."""
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return model


def generate_samples(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    num_samples: int = 20
) -> List[Dict]:
    """Generate sample responses."""
    samples = []
    
    for prompt in tqdm(prompts[:num_samples], desc="Generating samples"):
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=256
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
        
        samples.append({
            'prompt': prompt,
            'response': response
        })
    
    return samples


def load_training_history(output_dir: str) -> Dict:
    """Load training history."""
    history_path = os.path.join(output_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return json.load(f)
    return {}


def compare_all_methods(
    ppo_dir: str,
    grpo_dir: str,
    dpo_dir: str,
    output_dir: str = "./comparison_results"
):
    """Compare PPO, GRPO, and DPO."""
    os.makedirs(output_dir, exist_ok=True)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Load training histories
    print("\nLoading training histories...")
    ppo_history = load_training_history(ppo_dir)
    grpo_history = load_training_history(grpo_dir)
    dpo_history = load_training_history(dpo_dir)
    
    # Load models
    print("\nLoading models...")
    base_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    base_model.eval()
    
    models = {'base': base_model}
    
    if os.path.exists(os.path.join(ppo_dir, "final")):
        models['ppo'] = load_model(os.path.join(ppo_dir, "final"), device)
    
    if os.path.exists(os.path.join(grpo_dir, "final")):
        models['grpo'] = load_model(os.path.join(grpo_dir, "final"), device)
    
    if os.path.exists(os.path.join(dpo_dir, "final")):
        models['dpo'] = load_model(os.path.join(dpo_dir, "final"), device)
    
    # Load test prompts
    print("\nLoading test prompts...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    test_prompts = []
    for item in dataset:
        text = item['chosen']
        parts = text.split("Assistant:")
        if len(parts) >= 2:
            prompt = "Assistant:".join(parts[:-1]).strip()
            if prompt:
                test_prompts.append(prompt + "\n\nAssistant:")
    test_prompts = test_prompts[:50]
    
    # Generate samples from each model
    print("\nGenerating samples...")
    all_samples = {}
    for name, model in models.items():
        print(f"Generating from {name}...")
        all_samples[name] = generate_samples(model, tokenizer, test_prompts, device, num_samples=20)
    
    # Create comparison report
    results = {
        'training_summary': {},
        'efficiency': {}
    }
    
    # PPO summary
    if ppo_history:
        results['training_summary']['ppo'] = {
            'final_reward': float(np.mean(ppo_history.get('rewards', [0])[-50:])),
            'final_kl': float(np.mean(ppo_history.get('kl_divergence', [0])[-50:])),
            'total_steps': len(ppo_history.get('rewards', []))
        }
    
    # GRPO summary
    if grpo_history:
        results['training_summary']['grpo'] = {
            'final_reward': float(np.mean(grpo_history.get('rewards', [0])[-50:])),
            'final_kl': float(np.mean(grpo_history.get('kl_divergence', [0])[-50:])),
            'total_steps': len(grpo_history.get('rewards', []))
        }
    
    # DPO summary
    if dpo_history:
        results['training_summary']['dpo'] = {
            'final_loss': float(np.mean(dpo_history.get('loss', [0])[-50:])),
            'final_accuracy': float(np.mean(dpo_history.get('accuracies', [0])[-50:])),
            'final_margin': float(np.mean(dpo_history.get('reward_margins', [0])[-50:])),
            'total_steps': len(dpo_history.get('loss', []))
        }
    
    # Load efficiency stats
    for method, method_dir in [('ppo', ppo_dir), ('grpo', grpo_dir), ('dpo', dpo_dir)]:
        stats_path = os.path.join(method_dir, "efficiency_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                results['efficiency'][method] = json.load(f)
    
    # Save results
    results_path = os.path.join(output_dir, "comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save samples
    samples_path = os.path.join(output_dir, "generated_samples.json")
    with open(samples_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    # Create comparison plots
    create_comparison_plots(ppo_history, grpo_history, dpo_history, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY: PPO vs GRPO vs DPO")
    print("="*60)
    
    print("\n--- Training Results ---")
    if 'ppo' in results['training_summary']:
        ppo = results['training_summary']['ppo']
        print(f"PPO:  Final Reward = {ppo['final_reward']:.4f}, KL = {ppo['final_kl']:.4f}")
    
    if 'grpo' in results['training_summary']:
        grpo = results['training_summary']['grpo']
        print(f"GRPO: Final Reward = {grpo['final_reward']:.4f}, KL = {grpo['final_kl']:.4f}")
    
    if 'dpo' in results['training_summary']:
        dpo = results['training_summary']['dpo']
        print(f"DPO:  Final Loss = {dpo['final_loss']:.4f}, Accuracy = {dpo['final_accuracy']:.4f}, Margin = {dpo['final_margin']:.4f}")
    
    print("\n--- Efficiency ---")
    for method in ['ppo', 'grpo', 'dpo']:
        if method in results['efficiency']:
            eff = results['efficiency'][method]
            total_time = eff.get('avg_total_time', eff.get('avg_generation_time', 0) + eff.get('avg_forward_time', 0) + eff.get('avg_backward_time', 0))
            print(f"{method.upper()}: {total_time:.4f}s per step")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


def create_comparison_plots(
    ppo_history: Dict,
    grpo_history: Dict,
    dpo_history: Dict,
    output_dir: str
):
    """Create comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Rewards comparison (PPO vs GRPO)
    if ppo_history.get('rewards'):
        axes[0, 0].plot(ppo_history['rewards'], alpha=0.7, label='PPO')
    if grpo_history.get('rewards'):
        axes[0, 0].plot(grpo_history['rewards'], alpha=0.7, label='GRPO')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards: PPO vs GRPO')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: DPO Loss and Accuracy
    if dpo_history.get('loss'):
        ax2 = axes[0, 1]
        ax2.plot(dpo_history['loss'], alpha=0.7, color='blue', label='Loss')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Loss', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        if dpo_history.get('accuracies'):
            ax2_twin = ax2.twinx()
            ax2_twin.plot(dpo_history['accuracies'], alpha=0.7, color='green', label='Accuracy')
            ax2_twin.set_ylabel('Accuracy', color='green')
            ax2_twin.tick_params(axis='y', labelcolor='green')
            ax2_twin.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        
        ax2.set_title('DPO: Loss and Accuracy')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: KL Divergence comparison
    if ppo_history.get('kl_divergence'):
        axes[1, 0].plot(ppo_history['kl_divergence'], alpha=0.7, label='PPO')
    if grpo_history.get('kl_divergence'):
        axes[1, 0].plot(grpo_history['kl_divergence'], alpha=0.7, label='GRPO')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence: PPO vs GRPO')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: DPO Reward Margins
    if dpo_history.get('reward_margins'):
        axes[1, 1].plot(dpo_history['reward_margins'], alpha=0.7, color='purple')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Margin (Chosen - Rejected)')
        axes[1, 1].set_title('DPO: Reward Margin')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "method_comparison.png"), dpi=150)
    plt.close()
    print(f"Comparison plots saved to {output_dir}/method_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Compare PPO, GRPO, and DPO")
    parser.add_argument("--ppo_dir", type=str, default="../part2/ppo_output")
    parser.add_argument("--grpo_dir", type=str, default="../part2/grpo_output")
    parser.add_argument("--dpo_dir", type=str, default="./dpo_output")
    parser.add_argument("--output_dir", type=str, default="./comparison_results")
    
    args = parser.parse_args()
    
    compare_all_methods(
        args.ppo_dir,
        args.grpo_dir,
        args.dpo_dir,
        args.output_dir
    )


if __name__ == "__main__":
    main()