"""
Part 1.2: Reward Model Training
Train the reward model using pairwise ranking loss.
Tracks: accuracy, loss curves, gradient norms
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from reward_model import RewardModel, get_device


class PreferenceDataset(Dataset):
    """Dataset for preference pairs."""
    
    def __init__(
        self,
        data,
        tokenizer,
        max_length: int = 512
    ):
        """
        Initialize the preference dataset.
        
        Args:
            data: HuggingFace dataset split
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get the full chosen and rejected texts
        chosen_text = item['chosen_full'] if 'chosen_full' in item else item['chosen']
        rejected_text = item['rejected_full'] if 'rejected_full' in item else item['rejected']
        
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
            'chosen_text': chosen_text,
            'rejected_text': rejected_text
        }


class RewardModelTrainer:
    """Trainer for the Reward Model."""
    
    def __init__(
        self,
        model: RewardModel,
        tokenizer,
        device: torch.device,
        output_dir: str = "./reward_model_output",
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        use_fp16: bool = False
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Reward model to train
            tokenizer: Tokenizer
            device: Device to train on
            output_dir: Directory to save outputs
            learning_rate: Learning rate
            weight_decay: Weight decay for AdamW
            warmup_ratio: Ratio of warmup steps
            max_grad_norm: Maximum gradient norm for clipping
            use_fp16: Whether to use mixed precision training (CUDA only)
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        
        # Mixed precision training (only for CUDA)
        self.use_fp16 = use_fp16 and device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_fp16 else None
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'gradient_norms': [],
            'learning_rates': []
        }
        
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        eval_steps: int = 500,
        save_steps: int = 1000,
        logging_steps: int = 100
    ):
        """
        Train the reward model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of training epochs
            batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
        """
        # Create dataloaders - optimize for GPU
        use_cuda = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if use_cuda else 0,  # Use multiple workers for CUDA
            pin_memory=use_cuda,  # Pin memory for faster GPU transfer
            drop_last=True  # Drop incomplete batches for stability
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if use_cuda else 0,
            pin_memory=use_cuda
        )
        
        # Calculate total steps
        total_steps = len(train_loader) * epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print("\n" + "="*60)
        print("REWARD MODEL TRAINING")
        print("="*60)
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision (FP16): {self.use_fp16}")
        print("="*60 + "\n")
        
        # Training loop
        global_step = 0
        best_val_accuracy = 0.0
        
        self.model.train()
        
        for epoch in range(epochs):
            print(f"\n{'='*20} Epoch {epoch + 1}/{epochs} {'='*20}")
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            optimizer.zero_grad()
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                
                # Forward pass with mixed precision
                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        loss, chosen_rewards, rejected_rewards = self.model.compute_pairwise_loss(
                            chosen_input_ids,
                            chosen_attention_mask,
                            rejected_input_ids,
                            rejected_attention_mask
                        )
                        loss = loss / gradient_accumulation_steps
                    
                    # Backward pass with scaler
                    self.scaler.scale(loss).backward()
                else:
                    # Standard forward pass
                    loss, chosen_rewards, rejected_rewards = self.model.compute_pairwise_loss(
                        chosen_input_ids,
                        chosen_attention_mask,
                        rejected_input_ids,
                        rejected_attention_mask
                    )
                    loss = loss / gradient_accumulation_steps
                    loss.backward()
                
                # Track accuracy (chosen reward > rejected reward)
                with torch.no_grad():
                    correct = (chosen_rewards > rejected_rewards).sum().item()
                    epoch_correct += correct
                    epoch_total += chosen_rewards.size(0)
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                # Gradient accumulation
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Compute gradient norm before clipping
                    grad_norm = self._compute_gradient_norm()
                    self.history['gradient_norms'].append(grad_norm)
                    
                    if self.use_fp16:
                        # Unscale gradients and clip
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        # Update with scaler
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                        # Update weights
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Record learning rate
                    current_lr = scheduler.get_last_lr()[0]
                    self.history['learning_rates'].append(current_lr)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                        'acc': f"{epoch_correct/epoch_total:.4f}",
                        'grad_norm': f"{grad_norm:.4f}",
                        'lr': f"{current_lr:.2e}"
                    })
                    
                    # Logging
                    if global_step % logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        avg_acc = epoch_correct / epoch_total
                        self.history['train_loss'].append(avg_loss)
                        self.history['train_accuracy'].append(avg_acc)
                    
                    # Evaluation
                    if global_step % eval_steps == 0:
                        val_loss, val_acc = self.evaluate(val_loader)
                        self.history['val_loss'].append(val_loss)
                        self.history['val_accuracy'].append(val_acc)
                        
                        print(f"\n[Step {global_step}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                        
                        # Save best model
                        if val_acc > best_val_accuracy:
                            best_val_accuracy = val_acc
                            self._save_checkpoint("best_model.pt")
                            print(f"New best model saved! Accuracy: {val_acc:.4f}")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self._save_checkpoint(f"checkpoint_step_{global_step}.pt")
            
            # End of epoch evaluation
            val_loss, val_acc = self.evaluate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {epoch_loss / len(train_loader):.4f}")
            print(f"  Train Accuracy: {epoch_correct / epoch_total:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                self._save_checkpoint("best_model.pt")
                print(f"New best model saved! Accuracy: {val_acc:.4f}")
            
            self.model.train()
        
        # Save final model and training history
        self._save_checkpoint("final_model.pt")
        self._save_history()
        self._plot_training_curves()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
        print(f"Models saved to: {self.output_dir}")
        print("="*60)
        
        return best_val_accuracy
    
    def evaluate(self, dataloader: DataLoader) -> tuple:
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                
                loss, chosen_rewards, rejected_rewards = self.model.compute_pairwise_loss(
                    chosen_input_ids,
                    chosen_attention_mask,
                    rejected_input_ids,
                    rejected_attention_mask
                )
                
                total_loss += loss.item() * chosen_input_ids.size(0)
                total_correct += (chosen_rewards > rejected_rewards).sum().item()
                total_samples += chosen_input_ids.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def _compute_gradient_norm(self) -> float:
        """Compute the total gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        self.model.save_pretrained(path)
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Loss curves
        if self.history['train_loss']:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        if self.history['val_loss']:
            # Scale x-axis for val_loss to match train_loss
            val_x = np.linspace(0, len(self.history['train_loss']), len(self.history['val_loss']))
            axes[0, 0].plot(val_x, self.history['val_loss'], label='Val Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Logging Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        if self.history['train_accuracy']:
            axes[0, 1].plot(self.history['train_accuracy'], label='Train Acc', alpha=0.8)
        if self.history['val_accuracy']:
            val_x = np.linspace(0, len(self.history['train_accuracy']), len(self.history['val_accuracy']))
            axes[0, 1].plot(val_x, self.history['val_accuracy'], label='Val Acc', alpha=0.8)
        axes[0, 1].set_xlabel('Logging Steps')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Gradient norms
        if self.history['gradient_norms']:
            axes[1, 0].plot(self.history['gradient_norms'], alpha=0.7)
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].set_title('Gradient Norms During Training')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning rate
        if self.history['learning_rates']:
            axes[1, 1].plot(self.history['learning_rates'], alpha=0.7)
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Training curves saved to: {plot_path}")


def load_and_prepare_data(
    processed_data_path: str = None,
    max_samples: int = None,
    train_ratio: float = 0.9
):
    """
    Load and prepare data for training.
    
    Args:
        processed_data_path: Path to preprocessed data (if available)
        max_samples: Maximum samples to use
        train_ratio: Train/val split ratio
        
    Returns:
        train_data, val_data
    """
    if processed_data_path and os.path.exists(processed_data_path):
        print(f"Loading preprocessed data from: {processed_data_path}")
        dataset = load_from_disk(processed_data_path)
        train_data = dataset['train']
        val_data = dataset['validation']
    else:
        print("Loading raw data from HuggingFace...")
        dataset = load_dataset("Anthropic/hh-rlhf")
        
        # Use train split and create our own validation split
        all_data = dataset['train']
        
        if max_samples:
            all_data = all_data.select(range(min(max_samples, len(all_data))))
        
        # Split into train and validation
        split = all_data.train_test_split(test_size=1-train_ratio, seed=42)
        train_data = split['train']
        val_data = split['test']
    
    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_size = int(len(train_data) * (1 - train_ratio) / train_ratio)
        val_data = val_data.select(range(min(val_size, len(val_data))))
    
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Backbone model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_samples", type=int, default=30000, help="Maximum training samples")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16-32 for GPU, 4-8 for MPS)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./reward_model_output", help="Output directory")
    parser.add_argument("--processed_data_path", type=str, default=None, help="Path to preprocessed data")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training (CUDA only)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("REWARD MODEL TRAINING - Part 1.2")
    print("="*60)
    
    # Get device
    device = get_device()
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("\nLoading data...")
    train_data, val_data = load_and_prepare_data(
        processed_data_path=args.processed_data_path,
        max_samples=args.max_samples
    )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = PreferenceDataset(train_data, tokenizer, args.max_length)
    val_dataset = PreferenceDataset(val_data, tokenizer, args.max_length)
    
    # Initialize model
    print(f"\nInitializing Reward Model with backbone: {args.model_name}")
    model = RewardModel(model_name=args.model_name)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = RewardModelTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        use_fp16=args.fp16
    )
    
    # Train
    best_accuracy = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps
    )
    
    print(f"\nTraining complete! Best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()