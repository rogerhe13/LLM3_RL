"""
Part 3: Direct Preference Optimization (DPO) Implementation

DPO bypasses explicit reward modeling by directly optimizing the policy
using preference data.

Key formula:
L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

where:
- y_w = chosen/winning response
- y_l = rejected/losing response
- β = temperature parameter controlling deviation from reference
- π = policy model
- π_ref = reference model (frozen)
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


@dataclass
class DPOConfig:
    """Configuration for DPO training."""
    # Model
    model_name: str = "gpt2"
    
    # Training
    learning_rate: float = 5e-7
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_length: int = 512
    
    # DPO specific
    beta: float = 0.1  # Temperature parameter (controls KL penalty strength)
    
    # Training control
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    
    # Logging
    output_dir: str = "./dpo_output"
    logging_steps: int = 50
    save_steps: int = 200
    eval_steps: int = 100


class PreferenceDataset(Dataset):
    """Dataset for DPO training with preference pairs."""
    
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 512
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get chosen and rejected texts
        chosen_text = item['chosen']
        rejected_text = item['rejected']
        
        # Tokenize chosen
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize rejected
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0),
        }


class DPOTrainer:
    """
    DPO Trainer for direct preference optimization.
    
    DPO loss:
    L = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
    """
    
    def __init__(
        self,
        config: DPOConfig,
        tokenizer,
        device: torch.device
    ):
        self.config = config
        self.device = device
        self.tokenizer = tokenizer
        
        # Load policy model
        print(f"Loading policy model: {config.model_name}")
        self.policy = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
        
        # Load reference model (frozen copy)
        print(f"Loading reference model: {config.model_name}")
        self.reference = AutoModelForCausalLM.from_pretrained(config.model_name).to(device)
        self.reference.eval()
        for param in self.reference.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = AdamW(
            self.policy.parameters(),
            lr=config.learning_rate
        )
        
        # Output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'loss': [],
            'chosen_rewards': [],
            'rejected_rewards': [],
            'reward_margins': [],
            'accuracies': []
        }
        
        # Timing
        self.timing = {
            'forward_time': [],
            'backward_time': [],
            'total_time': []
        }
    
    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities.
        
        Args:
            model: Language model
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            log_probs: [batch_size] - sum of log probs per sequence
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # Mask padding and sum
        masked_log_probs = token_log_probs * shift_mask.float()
        sequence_log_probs = masked_log_probs.sum(dim=-1)  # [batch_size]
        
        return sequence_log_probs
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute DPO loss.
        
        L = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
          = -E[log σ(β * ((log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))))]
        
        Args:
            policy_chosen_logps: Log probs of chosen under policy
            policy_rejected_logps: Log probs of rejected under policy
            reference_chosen_logps: Log probs of chosen under reference
            reference_rejected_logps: Log probs of rejected under reference
            
        Returns:
            loss: Scalar DPO loss
            chosen_rewards: Implicit rewards for chosen
            rejected_rewards: Implicit rewards for rejected
        """
        # Compute log ratios (implicit rewards)
        # r(x, y) = β * log(π(y|x) / π_ref(y|x))
        chosen_rewards = self.config.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.config.beta * (policy_rejected_logps - reference_rejected_logps)
        
        # DPO loss: -log σ(r_chosen - r_rejected)
        logits = chosen_rewards - rejected_rewards
        
        # Numerical stability
        logits = torch.clamp(logits, -50.0, 50.0)
        
        loss = -F.logsigmoid(logits).mean()
        
        return loss, chosen_rewards, rejected_rewards
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single DPO training step.
        """
        t_start = time.time()
        
        # Move to device
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        
        # Get policy log probs
        t_fwd = time.time()
        policy_chosen_logps = self.get_log_probs(
            self.policy, chosen_input_ids, chosen_attention_mask
        )
        policy_rejected_logps = self.get_log_probs(
            self.policy, rejected_input_ids, rejected_attention_mask
        )
        
        # Get reference log probs (no gradient)
        with torch.no_grad():
            reference_chosen_logps = self.get_log_probs(
                self.reference, chosen_input_ids, chosen_attention_mask
            )
            reference_rejected_logps = self.get_log_probs(
                self.reference, rejected_input_ids, rejected_attention_mask
            )
        
        fwd_time = time.time() - t_fwd
        
        # Compute DPO loss
        loss, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        
        # Check for nan/inf
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: nan/inf loss detected, skipping batch")
            return {
                'loss': 0.0,
                'chosen_reward': 0.0,
                'rejected_reward': 0.0,
                'reward_margin': 0.0,
                'accuracy': 0.5
            }
        
        # Backward
        t_bwd = time.time()
        loss.backward()
        bwd_time = time.time() - t_bwd
        
        total_time = time.time() - t_start
        
        # Compute accuracy (how often chosen > rejected)
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
        
        # Record timing
        self.timing['forward_time'].append(fwd_time)
        self.timing['backward_time'].append(bwd_time)
        self.timing['total_time'].append(total_time)
        
        return {
            'loss': loss.item(),
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
            'accuracy': accuracy
        }
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ):
        """
        Full DPO training loop.
        """
        # Create dataloader
        use_cuda = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4 if use_cuda else 0,
            pin_memory=use_cuda,
            drop_last=True
        )
        
        # Calculate total steps
        total_steps = len(train_loader) * self.config.epochs // self.config.gradient_accumulation_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print("\n" + "="*60)
        print("DPO TRAINING")
        print("="*60)
        print(f"Training samples: {len(train_dataset)}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"Total steps: {total_steps}")
        print(f"Beta (temperature): {self.config.beta}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")
        
        global_step = 0
        best_accuracy = 0.0
        
        self.policy.train()
        
        for epoch in range(self.config.epochs):
            print(f"\n{'='*20} Epoch {epoch + 1}/{self.config.epochs} {'='*20}")
            
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            self.optimizer.zero_grad()
            progress_bar = tqdm(train_loader, desc=f"DPO Training Epoch {epoch + 1}")
            
            for step, batch in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                epoch_accuracy += metrics['accuracy']
                num_batches += 1
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Update
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Record history
                    self.history['loss'].append(metrics['loss'])
                    self.history['chosen_rewards'].append(metrics['chosen_reward'])
                    self.history['rejected_rewards'].append(metrics['rejected_reward'])
                    self.history['reward_margins'].append(metrics['reward_margin'])
                    self.history['accuracies'].append(metrics['accuracy'])
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'acc': f"{metrics['accuracy']:.3f}",
                        'margin': f"{metrics['reward_margin']:.3f}"
                    })
                    
                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        avg_loss = np.mean(self.history['loss'][-self.config.logging_steps:])
                        avg_acc = np.mean(self.history['accuracies'][-self.config.logging_steps:])
                        avg_margin = np.mean(self.history['reward_margins'][-self.config.logging_steps:])
                        print(f"\n[Step {global_step}] Loss: {avg_loss:.4f}, "
                              f"Acc: {avg_acc:.3f}, Margin: {avg_margin:.3f}")
                    
                    # Save checkpoint
                    if global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"checkpoint_step_{global_step}")
                        
                        # Check if best
                        current_acc = np.mean(self.history['accuracies'][-self.config.logging_steps:])
                        if current_acc > best_accuracy:
                            best_accuracy = current_acc
                            self.save_checkpoint("best")
                            print(f"New best model! Accuracy: {best_accuracy:.4f}")
            
            # End of epoch summary
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_acc = epoch_accuracy / num_batches
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Average Accuracy: {avg_epoch_acc:.4f}")
        
        # Save final model
        self.save_checkpoint("final")
        self.save_history()
        self.plot_training_curves()
        self.print_efficiency_stats()
        
        print("\n" + "="*60)
        print("DPO TRAINING COMPLETE!")
        print(f"Final average loss: {np.mean(self.history['loss'][-100:]):.4f}")
        print(f"Final average accuracy: {np.mean(self.history['accuracies'][-100:]):.4f}")
        print(f"Final average margin: {np.mean(self.history['reward_margins'][-100:]):.4f}")
        print("="*60)
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.policy.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                
                # Get log probs
                policy_chosen_logps = self.get_log_probs(
                    self.policy, chosen_input_ids, chosen_attention_mask
                )
                policy_rejected_logps = self.get_log_probs(
                    self.policy, rejected_input_ids, rejected_attention_mask
                )
                reference_chosen_logps = self.get_log_probs(
                    self.reference, chosen_input_ids, chosen_attention_mask
                )
                reference_rejected_logps = self.get_log_probs(
                    self.reference, rejected_input_ids, rejected_attention_mask
                )
                
                # Compute loss
                loss, chosen_rewards, rejected_rewards = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps
                )
                
                accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
                
                total_loss += loss.item()
                total_accuracy += accuracy
                num_batches += 1
        
        self.policy.train()
        
        return {
            'val_loss': total_loss / num_batches,
            'val_accuracy': total_accuracy / num_batches
        }
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        self.policy.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def save_history(self):
        """Save training history."""
        history_path = os.path.join(self.config.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def print_efficiency_stats(self):
        """Print efficiency statistics."""
        print("\n" + "="*60)
        print("EFFICIENCY STATISTICS")
        print("="*60)
        print(f"Average forward time: {np.mean(self.timing['forward_time']):.4f}s")
        print(f"Average backward time: {np.mean(self.timing['backward_time']):.4f}s")
        print(f"Average total time per step: {np.mean(self.timing['total_time']):.4f}s")
        
        # Save stats
        stats = {
            'avg_forward_time': float(np.mean(self.timing['forward_time'])),
            'avg_backward_time': float(np.mean(self.timing['backward_time'])),
            'avg_total_time': float(np.mean(self.timing['total_time']))
        }
        
        stats_path = os.path.join(self.config.output_dir, "efficiency_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def plot_training_curves(self):
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(self.history['loss'], alpha=0.7)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('DPO Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history['accuracies'], alpha=0.7, color='green')
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Preference Accuracy (Chosen > Rejected)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rewards
        axes[1, 0].plot(self.history['chosen_rewards'], alpha=0.7, label='Chosen', color='blue')
        axes[1, 0].plot(self.history['rejected_rewards'], alpha=0.7, label='Rejected', color='red')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Implicit Reward')
        axes[1, 0].set_title('Implicit Rewards (β * log ratio)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward margin
        axes[1, 1].plot(self.history['reward_margins'], alpha=0.7, color='purple')
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Margin')
        axes[1, 1].set_title('Reward Margin (Chosen - Rejected)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, "training_curves.png"), dpi=150)
        plt.close()
        print(f"Training curves saved to {self.config.output_dir}/training_curves.png")


def main():
    parser = argparse.ArgumentParser(description="DPO Training")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./dpo_output")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature parameter")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    
    # Config
    config = DPOConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps
    )
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Important for decoder-only models
    
    # Load data
    print("Loading training data...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    # Split into train/val
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split['train']
    val_data = split['test']
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = PreferenceDataset(train_data, tokenizer, config.max_length)
    val_dataset = PreferenceDataset(val_data, tokenizer, config.max_length)
    
    # Initialize trainer
    trainer = DPOTrainer(
        config=config,
        tokenizer=tokenizer,
        device=device
    )
    
    # Train
    trainer.train(train_dataset, val_dataset)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0
    )
    val_metrics = trainer.evaluate(val_loader)
    print(f"Final Validation Loss: {val_metrics['val_loss']:.4f}")
    print(f"Final Validation Accuracy: {val_metrics['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
