"""
Part 2.2: GRPO (Group Relative Policy Optimization) Implementation

Key differences from PPO:
- Samples multiple responses per prompt (group)
- Computes advantages relative to group mean
- Simplified policy gradient (no value function needed)
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from policy_model import PolicyModel, ReferenceModel, get_device

import sys
sys.path.append('../part1')
from reward_model import RewardModel


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""
    # Model
    model_name: str = "gpt2"
    reward_model_path: str = "./reward_model_output/best_model.pt"
    
    # Training
    learning_rate: float = 1e-6
    batch_size: int = 2              # Number of prompts per batch
    group_size: int = 4              # Number of responses per prompt
    gradient_accumulation_steps: int = 2
    max_length: int = 512
    max_new_tokens: int = 128
    
    # GRPO specific
    kl_coef: float = 0.1             # KL penalty coefficient
    clip_ratio: float = 0.2          # Optional clipping
    normalize_advantages: bool = True
    
    # Training control
    max_grad_norm: float = 1.0
    
    # Sampling
    temperature: float = 1.0
    top_p: float = 0.9
    
    # Logging
    output_dir: str = "./grpo_output"
    logging_steps: int = 10
    save_steps: int = 100


class GRPOTrainer:
    """
    GRPO Trainer for RLHF.
    
    Key idea: For each prompt, sample multiple responses and compute
    advantages relative to the group mean reward.
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        policy_model: PolicyModel,
        reward_model: RewardModel,
        tokenizer,
        device: torch.device
    ):
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        
        # Models
        self.policy = policy_model.to(device)
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        
        # Create reference model
        self.reference = ReferenceModel(self.policy)
        self.reference.model.to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'rewards': [],
            'kl_divergence': [],
            'policy_loss': [],
            'advantages': [],
            'total_loss': []
        }
        
        # Timing for efficiency comparison
        self.timing = {
            'generation_time': [],
            'forward_time': [],
            'backward_time': []
        }
    
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get rewards from reward model."""
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        return rewards.squeeze(-1)
    
    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int
    ) -> torch.Tensor:
        """
        Compute advantages relative to group mean.
        
        Args:
            rewards: [batch_size * group_size]
            group_size: Number of responses per prompt
            
        Returns:
            advantages: [batch_size * group_size]
        """
        # Reshape to [batch_size, group_size]
        batch_size = rewards.size(0) // group_size
        rewards_grouped = rewards.view(batch_size, group_size)
        
        # Compute mean reward per group
        group_mean = rewards_grouped.mean(dim=1, keepdim=True)
        
        # Advantages = reward - group_mean
        advantages = rewards_grouped - group_mean
        
        # Normalize if configured
        if self.config.normalize_advantages:
            std = advantages.std() + 1e-8
            advantages = advantages / std
        
        # Flatten back
        return advantages.view(-1)
    
    def generate_group_responses(
        self,
        prompts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        """
        Generate multiple responses per prompt.
        
        Args:
            prompts: List of prompts [batch_size]
            
        Returns:
            input_ids: [batch_size * group_size, seq_len]
            attention_mask: [batch_size * group_size, seq_len]
            response_mask: [batch_size * group_size, seq_len]
            prompt_lengths: List of prompt lengths
        """
        group_size = self.config.group_size
        
        # Repeat each prompt group_size times
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * group_size)
        
        # Tokenize
        prompt_encodings = self.tokenizer(
            expanded_prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length - self.config.max_new_tokens,
            return_tensors='pt'
        ).to(self.device)
        
        prompt_ids = prompt_encodings['input_ids']
        prompt_mask = prompt_encodings['attention_mask']
        prompt_lengths = prompt_mask.sum(dim=1).tolist()
        
        # Generate responses (with sampling for diversity)
        self.policy.eval()
        with torch.no_grad():
            generated_ids = self.policy.generate(
                prompt_ids,
                prompt_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        self.policy.train()
        
        # Create masks
        attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        
        batch_total = len(expanded_prompts)
        seq_len = generated_ids.size(1)
        response_mask = torch.zeros_like(generated_ids)
        
        for i in range(batch_total):
            response_mask[i, prompt_lengths[i]:] = 1
        response_mask = response_mask * attention_mask
        
        return generated_ids, attention_mask, response_mask, prompt_lengths
    
    def grpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute GRPO policy gradient loss.
        
        L = -E[A * log π(a|s)]
        
        With optional clipping like PPO.
        """
        mask = response_mask[:, 1:].float()
        lengths = mask.sum(dim=-1).clamp(min=1)
        
        # Use MEAN log probs (normalized by response length) to prevent ratio explosion
        log_probs_mean = (log_probs * mask).sum(dim=-1) / lengths
        old_log_probs_mean = (old_log_probs * mask).sum(dim=-1) / lengths
        
        # Compute ratio for clipping with numerical stability
        log_ratio = log_probs_mean - old_log_probs_mean
        log_ratio = torch.clamp(log_ratio, -5.0, 5.0)  # Smaller clamp range
        ratio = torch.exp(log_ratio)
        
        # Clipped objective (optional, for stability)
        clip_ratio = self.config.clip_ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        # Loss
        loss_1 = ratio * advantages
        loss_2 = clipped_ratio * advantages
        policy_loss = -torch.min(loss_1, loss_2).mean()
        
        # Handle nan/inf
        if torch.isnan(policy_loss) or torch.isinf(policy_loss):
            policy_loss = torch.tensor(0.0, device=policy_loss.device, requires_grad=True)
        
        return policy_loss
    
    def kl_penalty(
        self,
        policy_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty (always non-negative).
        
        KL ≈ 0.5 * (log_ratio)^2
        """
        mask = response_mask[:, 1:].float()
        lengths = mask.sum(dim=-1).clamp(min=1)
        
        # Use MEAN log probs (normalized by response length)
        policy_mean = (policy_log_probs * mask).sum(dim=-1) / lengths
        reference_mean = (reference_log_probs * mask).sum(dim=-1) / lengths
        
        # Log ratio per sequence
        log_ratio = policy_mean - reference_mean
        
        # Squared log ratio (always non-negative)
        kl_per_seq = 0.5 * (log_ratio ** 2)
        
        result = kl_per_seq.mean()
        
        # Handle nan/inf
        if torch.isnan(result) or torch.isinf(result):
            result = torch.tensor(0.0, device=result.device)
        
        return result
    
    def train_step(
        self,
        prompts: List[str]
    ) -> Dict[str, float]:
        """
        Single GRPO training step.
        
        1. Generate group of responses per prompt
        2. Compute rewards
        3. Compute group-relative advantages
        4. Update policy
        """
        import time
        
        # 1. Generate group responses
        t0 = time.time()
        input_ids, attention_mask, response_mask, _ = self.generate_group_responses(prompts)
        gen_time = time.time() - t0
        self.timing['generation_time'].append(gen_time)
        
        # 2. Compute rewards
        rewards = self.compute_rewards(input_ids, attention_mask)
        
        # 3. Compute group-relative advantages
        advantages = self.compute_group_advantages(rewards, self.config.group_size)
        
        # 4. Get old log probs (before update) - use eval mode to disable dropout
        self.policy.eval()
        with torch.no_grad():
            old_log_probs = self.policy.get_per_token_log_probs(input_ids, attention_mask)
        
        # 5. Get reference log probs (eval mode)
        ref_log_probs = self.reference.get_per_token_log_probs(input_ids, attention_mask)
        
        # 6. Forward pass - back to train mode
        self.policy.train()
        t0 = time.time()
        new_log_probs = self.policy.get_per_token_log_probs(input_ids, attention_mask)
        fwd_time = time.time() - t0
        self.timing['forward_time'].append(fwd_time)
        
        # 7. Compute losses
        policy_loss = self.grpo_loss(new_log_probs, old_log_probs, advantages, response_mask)
        kl = self.kl_penalty(new_log_probs, ref_log_probs, response_mask)
        
        total_loss = policy_loss + self.config.kl_coef * kl
        
        # Skip update if loss is nan/inf
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Skipping update due to nan/inf loss")
            return {
                'reward': rewards.mean().item(),
                'policy_loss': 0.0,
                'kl_divergence': 0.0,
                'advantage': advantages.mean().item(),
                'total_loss': 0.0
            }
        
        # 8. Backward
        t0 = time.time()
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        bwd_time = time.time() - t0
        self.timing['backward_time'].append(bwd_time)
        
        return {
            'reward': rewards.mean().item(),
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl.item(),
            'advantage': advantages.mean().item(),
            'total_loss': total_loss.item()
        }
    
    def train(
        self,
        prompts: List[str],
        num_steps: int = 500
    ):
        """Full GRPO training loop."""
        print("\n" + "="*60)
        print("GRPO TRAINING")
        print("="*60)
        print(f"Number of prompts: {len(prompts)}")
        print(f"Training steps: {num_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Group size: {self.config.group_size}")
        print(f"Effective samples per step: {self.config.batch_size * self.config.group_size}")
        print(f"KL coefficient: {self.config.kl_coef}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        self.policy.train()
        
        progress_bar = tqdm(range(num_steps), desc="GRPO Training")
        
        for step in progress_bar:
            # Sample batch of prompts
            batch_indices = np.random.choice(
                len(prompts), 
                self.config.batch_size, 
                replace=False
            )
            batch_prompts = [prompts[i] for i in batch_indices]
            
            # Training step
            metrics = self.train_step(batch_prompts)
            
            # Record history
            self.history['rewards'].append(metrics['reward'])
            self.history['kl_divergence'].append(metrics['kl_divergence'])
            self.history['policy_loss'].append(metrics['policy_loss'])
            self.history['advantages'].append(metrics['advantage'])
            self.history['total_loss'].append(metrics['total_loss'])
            
            # Update progress bar
            progress_bar.set_postfix({
                'reward': f"{metrics['reward']:.3f}",
                'kl': f"{metrics['kl_divergence']:.4f}",
                'loss': f"{metrics['total_loss']:.4f}"
            })
            
            # Logging
            if (step + 1) % self.config.logging_steps == 0:
                avg_reward = np.mean(self.history['rewards'][-self.config.logging_steps:])
                avg_kl = np.mean(self.history['kl_divergence'][-self.config.logging_steps:])
                avg_gen_time = np.mean(self.timing['generation_time'][-self.config.logging_steps:])
                print(f"\n[Step {step + 1}] Avg Reward: {avg_reward:.4f}, "
                      f"Avg KL: {avg_kl:.4f}, Avg Gen Time: {avg_gen_time:.2f}s")
            
            # Save checkpoint
            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint_step_{step + 1}")
        
        # Save final
        self.save_checkpoint("final")
        self.save_history()
        self.plot_training_curves()
        self.print_efficiency_stats()
        
        print("\n" + "="*60)
        print("GRPO TRAINING COMPLETE!")
        print(f"Final average reward: {np.mean(self.history['rewards'][-100:]):.4f}")
        print(f"Final average KL: {np.mean(self.history['kl_divergence'][-100:]):.4f}")
        print("="*60)
    
    def print_efficiency_stats(self):
        """Print efficiency statistics for comparison with PPO."""
        print("\n" + "="*60)
        print("EFFICIENCY STATISTICS")
        print("="*60)
        print(f"Average generation time: {np.mean(self.timing['generation_time']):.3f}s")
        print(f"Average forward time: {np.mean(self.timing['forward_time']):.3f}s")
        print(f"Average backward time: {np.mean(self.timing['backward_time']):.3f}s")
        print(f"Total time per step: {np.mean(self.timing['generation_time']) + np.mean(self.timing['forward_time']) + np.mean(self.timing['backward_time']):.3f}s")
        
        # Save stats
        stats = {
            'avg_generation_time': float(np.mean(self.timing['generation_time'])),
            'avg_forward_time': float(np.mean(self.timing['forward_time'])),
            'avg_backward_time': float(np.mean(self.timing['backward_time']))
        }
        
        stats_path = os.path.join(self.config.output_dir, "efficiency_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        self.policy.save_pretrained(save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def save_history(self):
        """Save training history."""
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Rewards
        axes[0, 0].plot(self.history['rewards'], alpha=0.7)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards (GRPO)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL divergence
        axes[0, 1].plot(self.history['kl_divergence'], alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].set_title('KL Divergence from Reference')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Policy loss
        axes[1, 0].plot(self.history['policy_loss'], alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Advantages
        axes[1, 1].plot(self.history['advantages'], alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Advantage')
        axes[1, 1].set_title('Average Advantage')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "training_curves.png"), dpi=150)
        plt.close()
        print(f"Training curves saved to {self.config.output_dir}/training_curves.png")


def extract_prompts(dataset, max_samples: int = 5000) -> List[str]:
    """Extract prompts from HH-RLHF dataset."""
    prompts = []
    
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        
        text = item['chosen']
        parts = text.split("Assistant:")
        if len(parts) >= 2:
            prompt = "Assistant:".join(parts[:-1]).strip()
            if prompt:
                prompts.append(prompt + "\n\nAssistant:")
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for RLHF")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./grpo_output")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    
    # Config
    config = GRPOConfig(
        model_name=args.model_name,
        reward_model_path=args.reward_model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        max_new_tokens=args.max_new_tokens
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Important for decoder-only models
    
    # Load models
    print("Loading Policy Model...")
    policy = PolicyModel(args.model_name)
    
    print(f"Loading Reward Model from {args.reward_model_path}...")
    reward_model = RewardModel.load_pretrained(args.reward_model_path, device=device)
    
    # Load prompts
    print("Loading training prompts...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    prompts = extract_prompts(dataset, max_samples=args.max_samples)
    print(f"Loaded {len(prompts)} prompts")
    
    # Initialize trainer
    trainer = GRPOTrainer(
        config=config,
        policy_model=policy,
        reward_model=reward_model,
        tokenizer=tokenizer,
        device=device
    )
    
    # Train
    trainer.train(prompts, num_steps=args.num_steps)


if __name__ == "__main__":
    main()