"""
Simple SFT (Supervised Fine-Tuning) on HH-RLHF chosen responses.
This creates the initialization model for PPO/GRPO training.
"""

import os
import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SFTDataset(Dataset):
    """Dataset for SFT on chosen responses."""
    
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Use the chosen response as target
        text = self.data[idx]['chosen']
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Labels = input_ids (for causal LM)
        labels = input_ids.clone()
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


def train_sft(
    model_name: str = "gpt2",
    output_dir: str = "./sft_output",
    max_samples: int = 10000,
    epochs: int = 1,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512
):
    """Train SFT model on chosen responses."""
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Load data
    print("Loading data...")
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Training samples: {len(dataset)}")
    
    # Create dataset and dataloader
    train_dataset = SFTDataset(dataset, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda'
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    print("\n" + "="*60)
    print("SFT TRAINING")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Total steps: {total_steps}")
    print("="*60 + "\n")
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nSFT model saved to {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="SFT Training")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./sft_output")
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    
    args = parser.parse_args()
    
    train_sft(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()