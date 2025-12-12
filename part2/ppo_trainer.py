"""
Part 2.1: PPO-based RLHF Implementation

Implements:
- Clipped surrogate objective
- KL divergence penalty from reference policy  
- Entropy bonus for exploration
- Value function (critic) for advantage estimation
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from policy_model import PolicyModel, ReferenceModel, get_device

# Import reward model from part1
import sys
sys.path.append('../part1')
from reward_model import RewardModel


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Model
    model_name: str = "gpt2"
    reward_model_path: str = "./reward_model_output/best_model.pt"
    
    # Training
    learning_rate: float = 1e-6
    epochs: int = 1
    batch_size: int = 4
    mini_batch_size: int = 2
    gradient_accumulation_steps: int = 2
    max_length: int = 512
    max_new_tokens: int = 128
    
    # PPO specific
    clip_ratio: float = 0.2          # PPO clip ratio (epsilon)
    kl_coef: float = 0.1             # KL penalty coefficient (beta)
    entropy_coef: float = 0.01       # Entropy bonus coefficient
    gamma: float = 1.0               # Discount factor
    gae_lambda: float = 0.95         # GAE lambda
    
    # Value function
    vf_coef: float = 0.5             # Value function loss coefficient
    
    # Training control
    target_kl: float = 0.1           # Target KL for early stopping
    max_grad_norm: float = 1.0
    num_ppo_epochs: int = 4          # PPO epochs per batch
    
    # Sampling
    temperature: float = 1.0
    top_p: float = 0.9
    
    # Logging
    output_dir: str = "./ppo_output"
    logging_steps: int = 10
    save_steps: int = 100


class ValueHead(nn.Module):
    """Value function head for PPO."""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
        Returns:
            values: [batch_size, seq_len]
        """
        return self.value_head(hidden_states).squeeze(-1)


class PPOTrainer:
    """PPO Trainer for RLHF."""
    
    def __init__(
        self,
        config: PPOConfig,
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
        self.reward_model.eval()  # Freeze reward model
        
        # Create reference model (frozen copy)
        self.reference = ReferenceModel(self.policy)
        self.reference.model.to(device)
        
        # Value head
        hidden_size = self.policy.config.n_embd
        self.value_head = ValueHead(hidden_size).to(device)
        
        # Optimizer (policy + value head)
        self.optimizer = AdamW([
            {'params': self.policy.parameters(), 'lr': config.learning_rate},
            {'params': self.value_head.parameters(), 'lr': config.learning_rate * 10}
        ])
        
        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'rewards': [],
            'kl_divergence': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': []
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
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        response_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute advantages using GAE (Generalized Advantage Estimation).
        
        For sequence-level rewards, we simplify:
        - Advantage = Reward - Value (at the last response token)
        - Returns = Reward
        """
        batch_size = rewards.size(0)
        
        # Get value at the last response token
        response_lengths = response_mask.sum(dim=1).long() - 1
        last_values = values[torch.arange(batch_size, device=self.device), response_lengths]
        
        # Simple advantage: reward - baseline value
        advantages = rewards - last_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Returns for value function training
        returns = rewards
        
        return advantages, returns
    
    def ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PPO clipped surrogate objective.
        
        L_CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
        
        where r_t = π_θ(a|s) / π_θ_old(a|s) = exp(log_prob_new - log_prob_old)
        """
        # Compute probability ratio
        # Use MEAN log probs (normalized by response length) to avoid ratio explosion
        mask = response_mask[:, 1:].float()
        lengths = mask.sum(dim=-1).clamp(min=1)
        
        # Normalize by response length to prevent ratio explosion
        old_log_probs_mean = (old_log_probs * mask).sum(dim=-1) / lengths
        new_log_probs_mean = (new_log_probs * mask).sum(dim=-1) / lengths
        
        # Clamp log ratio (smaller range since we're using mean)
        log_ratio = new_log_probs_mean - old_log_probs_mean
        log_ratio = torch.clamp(log_ratio, -5.0, 5.0)  # Smaller clamp range
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate objective
        clip_ratio = self.config.clip_ratio
        clipped_ratio = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio)
        
        # Policy loss (negative because we want to maximize)
        policy_loss_1 = ratio * advantages
        policy_loss_2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Handle nan/inf
        if torch.isnan(policy_loss) or torch.isinf(policy_loss):
            policy_loss = torch.tensor(0.0, device=policy_loss.device)
        
        return policy_loss, ratio
    
    def entropy_loss(
        self,
        logits: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy bonus for exploration.
        
        H(π) = -Σ π(a|s) log π(a|s)
        """
        # Get probabilities
        probs = F.softmax(logits[:, :-1, :], dim=-1)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        
        # Entropy per token
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Mask to response tokens
        mask = response_mask[:, 1:].float()
        masked_entropy = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        
        return masked_entropy.mean()
    
    def value_loss(
        self,
        values: torch.Tensor,
        returns: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute value function loss (MSE).
        """
        batch_size = returns.size(0)
        
        # Get value at last response token
        response_lengths = response_mask.sum(dim=1).long() - 1
        last_values = values[torch.arange(batch_size, device=self.device), response_lengths]
        
        # MSE loss
        vf_loss = F.mse_loss(last_values, returns)
        
        return vf_loss
    
    def kl_penalty(
        self,
        policy_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence penalty.
        
        KL(π || π_ref) ≈ 0.5 * (log_ratio)^2  (approximation, always non-negative)
        """
        mask = response_mask[:, 1:].float()
        lengths = mask.sum(dim=-1).clamp(min=1)
        
        # Use MEAN log probs (normalized by response length)
        policy_mean = (policy_log_probs * mask).sum(dim=-1) / lengths
        reference_mean = (reference_log_probs * mask).sum(dim=-1) / lengths
        
        # Log ratio per sequence (already normalized)
        log_ratio = policy_mean - reference_mean
        
        # Squared log ratio (always non-negative)
        kl_per_seq = 0.5 * (log_ratio ** 2)
        
        result = kl_per_seq.mean()
        
        # Handle nan/inf
        if torch.isnan(result) or torch.isinf(result):
            result = torch.tensor(0.0, device=result.device)
        
        return result
    
    def generate_responses(
        self,
        prompts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate responses for a batch of prompts.
        
        Returns:
            input_ids: Full sequence (prompt + response)
            attention_mask: Attention mask
            response_mask: Mask indicating response tokens
        """
        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length - self.config.max_new_tokens,
            return_tensors='pt'
        ).to(self.device)
        
        prompt_ids = prompt_encodings['input_ids']
        prompt_mask = prompt_encodings['attention_mask']
        prompt_lengths = prompt_mask.sum(dim=1)
        
        # Generate responses
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
        
        # Create attention mask for full sequence
        attention_mask = (generated_ids != self.tokenizer.pad_token_id).long()
        
        # Create response mask (1 for response tokens, 0 for prompt)
        batch_size, seq_len = generated_ids.shape
        response_mask = torch.zeros_like(generated_ids)
        for i in range(batch_size):
            response_mask[i, prompt_lengths[i]:] = 1
        response_mask = response_mask * attention_mask  # Also mask padding
        
        return generated_ids, attention_mask, response_mask
    
    def train_step(
        self,
        prompts: List[str]
    ) -> Dict[str, float]:
        """
        Single PPO training step.
        
        1. Generate responses from current policy
        2. Compute rewards
        3. Compute advantages
        4. Update policy with PPO objective
        """
        # 1. Generate responses
        input_ids, attention_mask, response_mask = self.generate_responses(prompts)
        
        # 2. Get rewards
        rewards = self.compute_rewards(input_ids, attention_mask)
        
        # 3. Get old log probs and values (before update) - use eval mode to disable dropout
        self.policy.eval()
        with torch.no_grad():
            old_log_probs = self.policy.get_per_token_log_probs(input_ids, attention_mask)
            
            # Get hidden states for value function
            outputs = self.policy.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]
            old_values = self.value_head(hidden_states)
        
        # 4. Compute advantages
        advantages, returns = self.compute_advantages(rewards, old_values, response_mask)
        
        # 5. Get reference log probs (eval mode)
        ref_log_probs = self.reference.get_per_token_log_probs(input_ids, attention_mask)
        
        # 6. PPO update (multiple epochs)
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        
        self.policy.train()  # Back to train mode for gradient computation
        
        for _ in range(self.config.num_ppo_epochs):
            # Forward pass - get logits
            outputs = self.policy.model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1]
            
            # Get new log probs
            new_log_probs = self.policy.get_per_token_log_probs(input_ids, attention_mask)
            
            # Get new values
            new_values = self.value_head(hidden_states)
            
            # Compute losses
            policy_loss, ratio = self.ppo_loss(old_log_probs, new_log_probs, advantages, response_mask)
            vf_loss = self.value_loss(new_values, returns, response_mask)
            entropy = self.entropy_loss(logits, response_mask)
            kl = self.kl_penalty(new_log_probs, ref_log_probs, response_mask)
            
            # Total loss
            loss = (
                policy_loss 
                + self.config.vf_coef * vf_loss 
                - self.config.entropy_coef * entropy
                + self.config.kl_coef * kl
            )
            
            # Skip update if loss is nan/inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Skipping update due to nan/inf loss")
                continue
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.parameters()) + list(self.value_head.parameters()),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += vf_loss.item()
            total_entropy += entropy.item()
            total_kl += kl.item()
            
            # Early stopping if KL too high
            if kl.item() > self.config.target_kl * 1.5:
                break
        
        num_updates = self.config.num_ppo_epochs
        
        return {
            'reward': rewards.mean().item(),
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'kl_divergence': total_kl / num_updates,
            'total_loss': loss.item()
        }
    
    def train(
        self,
        prompts: List[str],
        num_steps: int = 1000
    ):
        """
        Full PPO training loop.
        
        Args:
            prompts: List of training prompts
            num_steps: Number of training steps
        """
        print("\n" + "="*60)
        print("PPO TRAINING")
        print("="*60)
        print(f"Number of prompts: {len(prompts)}")
        print(f"Training steps: {num_steps}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"PPO epochs per step: {self.config.num_ppo_epochs}")
        print(f"Clip ratio: {self.config.clip_ratio}")
        print(f"KL coefficient: {self.config.kl_coef}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        self.policy.train()
        
        progress_bar = tqdm(range(num_steps), desc="PPO Training")
        
        for step in progress_bar:
            # Sample batch of prompts
            batch_indices = np.random.choice(len(prompts), self.config.batch_size, replace=False)
            batch_prompts = [prompts[i] for i in batch_indices]
            
            # Training step
            metrics = self.train_step(batch_prompts)
            
            # Record history
            for key, value in metrics.items():
                if key in self.history:
                    self.history[key].append(value)
                elif key == 'reward':
                    self.history['rewards'].append(value)
            
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
                print(f"\n[Step {step + 1}] Avg Reward: {avg_reward:.4f}, Avg KL: {avg_kl:.4f}")
            
            # Save checkpoint
            if (step + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint_step_{step + 1}")
        
        # Save final model
        self.save_checkpoint("final")
        self.save_history()
        self.plot_training_curves()
        
        print("\n" + "="*60)
        print("PPO TRAINING COMPLETE!")
        print(f"Final average reward: {np.mean(self.history['rewards'][-100:]):.4f}")
        print(f"Final average KL: {np.mean(self.history['kl_divergence'][-100:]):.4f}")
        print("="*60)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        self.policy.save_pretrained(save_path)
        torch.save(self.value_head.state_dict(), os.path.join(save_path, "value_head.pt"))
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
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL divergence
        axes[0, 1].plot(self.history['kl_divergence'], alpha=0.7, color='orange')
        axes[0, 1].axhline(y=self.config.target_kl, color='r', linestyle='--', label='Target KL')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].set_title('KL Divergence from Reference')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Policy loss
        axes[1, 0].plot(self.history['policy_loss'], alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Policy Loss')
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        axes[1, 1].plot(self.history['entropy'], alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].set_title('Policy Entropy')
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
        
        # Parse the conversation to get the prompt
        text = item['chosen']
        
        # Split by last "Assistant:" to get prompt
        parts = text.split("Assistant:")
        if len(parts) >= 2:
            prompt = "Assistant:".join(parts[:-1]).strip()
            if prompt:
                prompts.append(prompt + "\n\nAssistant:")
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="PPO Training for RLHF")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ppo_output")
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    
    # Config
    config = PPOConfig(
        model_name=args.model_name,
        reward_model_path=args.reward_model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        clip_ratio=args.clip_ratio,
        kl_coef=args.kl_coef,
        entropy_coef=args.entropy_coef,
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
    trainer = PPOTrainer(
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
