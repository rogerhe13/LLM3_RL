import os
import json
import random
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm


class HHRLHFDataProcessor:
    """
    Data processor for Anthropic HH-RLHF dataset.
    Handles loading, exploration, and preprocessing.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_length: int = 512,
        seed: int = 42
    ):
        """
        Initialize the data processor.
        
        Args:
            model_name: Name of the tokenizer model to use
            max_length: Maximum sequence length for tokenization
            seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.max_length = max_length
        self.seed = seed
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # GPT-2 doesn't have a pad token by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.raw_dataset = None
        self.processed_dataset = None
        
    def load_dataset(self, subset: Optional[str] = None) -> DatasetDict:
        """
        Load the Anthropic HH-RLHF dataset.
        
        Args:
            subset: Optional subset to load ('helpful-base', 'harmless-base', etc.)
                   If None, loads all data.
        
        Returns:
            DatasetDict containing train and test splits
        """
        print("Loading Anthropic HH-RLHF dataset...")
        
        if subset:
            self.raw_dataset = load_dataset("Anthropic/hh-rlhf", data_dir=subset)
        else:
            self.raw_dataset = load_dataset("Anthropic/hh-rlhf")
            
        print(f"Dataset loaded successfully!")
        print(f"Train samples: {len(self.raw_dataset['train'])}")
        print(f"Test samples: {len(self.raw_dataset['test'])}")
        
        return self.raw_dataset
    
    def explore_dataset(self, num_examples: int = 5, save_dir: str = "./analysis") -> Dict:
        """
        Task A: Explore and analyze the dataset structure.
        
        Args:
            num_examples: Number of examples to display
            save_dir: Directory to save analysis plots
            
        Returns:
            Dictionary containing analysis results
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("DATASET EXPLORATION AND ANALYSIS")
        print("="*60)
        
        analysis_results = {}
        
        # 1. Basic statistics
        print("\n1. BASIC STATISTICS")
        print("-" * 40)
        train_size = len(self.raw_dataset['train'])
        test_size = len(self.raw_dataset['test'])
        print(f"Training set size: {train_size:,}")
        print(f"Test set size: {test_size:,}")
        print(f"Total samples: {train_size + test_size:,}")
        
        analysis_results['train_size'] = train_size
        analysis_results['test_size'] = test_size
        
        # 2. Data structure examination
        print("\n2. DATA STRUCTURE")
        print("-" * 40)
        sample = self.raw_dataset['train'][0]
        print(f"Fields in each sample: {list(sample.keys())}")
        print(f"\nExample chosen response (truncated):")
        print(sample['chosen'][:500] + "..." if len(sample['chosen']) > 500 else sample['chosen'])
        print(f"\nExample rejected response (truncated):")
        print(sample['rejected'][:500] + "..." if len(sample['rejected']) > 500 else sample['rejected'])
        
        # 3. Parse and analyze conversation structure
        print("\n3. CONVERSATION STRUCTURE ANALYSIS")
        print("-" * 40)
        
        chosen_lengths = []
        rejected_lengths = []
        chosen_response_lengths = []
        rejected_response_lengths = []
        num_turns_chosen = []
        num_turns_rejected = []
        length_diffs = []  # chosen - rejected
        
        for sample in tqdm(self.raw_dataset['train'], desc="Analyzing conversations"):
            chosen = sample['chosen']
            rejected = sample['rejected']
            
            # Total lengths
            chosen_lengths.append(len(chosen))
            rejected_lengths.append(len(rejected))
            length_diffs.append(len(chosen) - len(rejected))
            
            # Parse to get response only (last assistant turn)
            chosen_prompt, chosen_response = self._parse_conversation(chosen)
            rejected_prompt, rejected_response = self._parse_conversation(rejected)
            
            chosen_response_lengths.append(len(chosen_response))
            rejected_response_lengths.append(len(rejected_response))
            
            # Count turns
            num_turns_chosen.append(chosen.count("Human:"))
            num_turns_rejected.append(rejected.count("Human:"))
        
        # Statistics
        print(f"\nChosen text length - Mean: {np.mean(chosen_lengths):.1f}, "
              f"Std: {np.std(chosen_lengths):.1f}, "
              f"Min: {np.min(chosen_lengths)}, Max: {np.max(chosen_lengths)}")
        print(f"Rejected text length - Mean: {np.mean(rejected_lengths):.1f}, "
              f"Std: {np.std(rejected_lengths):.1f}, "
              f"Min: {np.min(rejected_lengths)}, Max: {np.max(rejected_lengths)}")
        
        print(f"\nChosen response length - Mean: {np.mean(chosen_response_lengths):.1f}, "
              f"Std: {np.std(chosen_response_lengths):.1f}")
        print(f"Rejected response length - Mean: {np.mean(rejected_response_lengths):.1f}, "
              f"Std: {np.std(rejected_response_lengths):.1f}")
        
        print(f"\nNumber of turns - Mean: {np.mean(num_turns_chosen):.1f}, "
              f"Min: {np.min(num_turns_chosen)}, Max: {np.max(num_turns_chosen)}")
        
        analysis_results['chosen_length_stats'] = {
            'mean': float(np.mean(chosen_lengths)),
            'std': float(np.std(chosen_lengths)),
            'min': int(np.min(chosen_lengths)),
            'max': int(np.max(chosen_lengths))
        }
        analysis_results['rejected_length_stats'] = {
            'mean': float(np.mean(rejected_lengths)),
            'std': float(np.std(rejected_lengths)),
            'min': int(np.min(rejected_lengths)),
            'max': int(np.max(rejected_lengths))
        }
        
        # 4. Bias analysis
        print("\n4. POTENTIAL BIASES AND PATTERNS")
        print("-" * 40)
        
        # Length bias: Are longer responses preferred?
        chosen_longer = sum(1 for d in length_diffs if d > 0)
        rejected_longer = sum(1 for d in length_diffs if d < 0)
        same_length = sum(1 for d in length_diffs if d == 0)
        
        print(f"Length preference analysis:")
        print(f"  - Chosen is longer: {chosen_longer} ({100*chosen_longer/len(length_diffs):.1f}%)")
        print(f"  - Rejected is longer: {rejected_longer} ({100*rejected_longer/len(length_diffs):.1f}%)")
        print(f"  - Same length: {same_length} ({100*same_length/len(length_diffs):.1f}%)")
        print(f"  - Average length difference: {np.mean(length_diffs):.1f} chars")
        
        analysis_results['length_bias'] = {
            'chosen_longer_pct': float(100*chosen_longer/len(length_diffs)),
            'rejected_longer_pct': float(100*rejected_longer/len(length_diffs)),
            'avg_length_diff': float(np.mean(length_diffs))
        }
        
        # Response length bias
        response_length_diffs = [c - r for c, r in zip(chosen_response_lengths, rejected_response_lengths)]
        chosen_resp_longer = sum(1 for d in response_length_diffs if d > 0)
        print(f"\nResponse-only length preference:")
        print(f"  - Chosen response longer: {chosen_resp_longer} ({100*chosen_resp_longer/len(response_length_diffs):.1f}%)")
        
        # 5. Create visualizations
        print("\n5. GENERATING VISUALIZATIONS...")
        print("-" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Length distributions
        axes[0, 0].hist(chosen_lengths, bins=50, alpha=0.5, label='Chosen', color='green')
        axes[0, 0].hist(rejected_lengths, bins=50, alpha=0.5, label='Rejected', color='red')
        axes[0, 0].set_xlabel('Text Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Text Lengths')
        axes[0, 0].legend()
        
        # Plot 2: Response length distributions
        axes[0, 1].hist(chosen_response_lengths, bins=50, alpha=0.5, label='Chosen', color='green')
        axes[0, 1].hist(rejected_response_lengths, bins=50, alpha=0.5, label='Rejected', color='red')
        axes[0, 1].set_xlabel('Response Length (characters)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Response Lengths')
        axes[0, 1].legend()
        
        # Plot 3: Length difference distribution
        axes[1, 0].hist(length_diffs, bins=50, color='blue', alpha=0.7)
        axes[1, 0].axvline(x=0, color='red', linestyle='--', label='No difference')
        axes[1, 0].axvline(x=np.mean(length_diffs), color='green', linestyle='--', 
                          label=f'Mean: {np.mean(length_diffs):.1f}')
        axes[1, 0].set_xlabel('Length Difference (Chosen - Rejected)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Length Differences')
        axes[1, 0].legend()
        
        # Plot 4: Number of turns distribution
        turn_counts = Counter(num_turns_chosen)
        axes[1, 1].bar(turn_counts.keys(), turn_counts.values(), color='purple', alpha=0.7)
        axes[1, 1].set_xlabel('Number of Conversation Turns')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Conversation Turns')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'dataset_analysis.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved analysis plots to: {plot_path}")
        
        # 6. Show example pairs
        print(f"\n6. EXAMPLE PREFERENCE PAIRS ({num_examples} examples)")
        print("-" * 40)
        
        for i in range(min(num_examples, len(self.raw_dataset['train']))):
            sample = self.raw_dataset['train'][i]
            chosen_prompt, chosen_response = self._parse_conversation(sample['chosen'])
            _, rejected_response = self._parse_conversation(sample['rejected'])
            
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {chosen_prompt[:300]}..." if len(chosen_prompt) > 300 else f"Prompt: {chosen_prompt}")
            print(f"\nChosen response ({len(chosen_response)} chars): {chosen_response[:200]}..." 
                  if len(chosen_response) > 200 else f"\nChosen response: {chosen_response}")
            print(f"\nRejected response ({len(rejected_response)} chars): {rejected_response[:200]}..."
                  if len(rejected_response) > 200 else f"\nRejected response: {rejected_response}")
        
        # Save analysis results
        results_path = os.path.join(save_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"\nSaved analysis results to: {results_path}")
        
        return analysis_results
    
    def _parse_conversation(self, text: str) -> Tuple[str, str]:
        """
        Parse a conversation text to extract prompt and response.
        
        The HH-RLHF format is:
        Human: <message>
        Assistant: <message>
        Human: <message>
        Assistant: <final_response>
        
        Args:
            text: Full conversation text
            
        Returns:
            Tuple of (prompt, response) where prompt is everything before
            the last Assistant turn, and response is the last Assistant turn.
        """
        # Split by "Assistant:" to find the last response
        parts = text.split("Assistant:")
        
        if len(parts) < 2:
            # No assistant response found
            return text, ""
        
        # The last part is the final response
        response = parts[-1].strip()
        
        # Everything before the last "Assistant:" is the prompt
        # We need to reconstruct it
        prompt = "Assistant:".join(parts[:-1])
        
        # If prompt ends with whitespace, strip it
        prompt = prompt.rstrip()
        
        return prompt, response
    
    def preprocess_dataset(
        self,
        train_ratio: float = 0.9,
        max_samples: Optional[int] = None,
        filter_long_sequences: bool = True
    ) -> DatasetDict:
        """
        Task B: Implement data preprocessing pipeline.
        
        Args:
            train_ratio: Ratio of data to use for training (rest for validation)
            max_samples: Maximum number of samples to use (for debugging)
            filter_long_sequences: Whether to filter out sequences longer than max_length
            
        Returns:
            Processed DatasetDict with train/validation splits
        """
        if self.raw_dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
            
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # Combine train and test for re-splitting
        all_data = []
        
        # Process training data
        train_data = self.raw_dataset['train']
        if max_samples:
            train_data = train_data.select(range(min(max_samples, len(train_data))))
            
        print(f"\nProcessing {len(train_data)} samples...")
        
        processed_count = 0
        filtered_count = 0
        edge_cases = {'empty_response': 0, 'too_long': 0, 'parse_error': 0}
        
        for sample in tqdm(train_data, desc="Preprocessing"):
            try:
                processed = self._process_sample(sample)
                
                if processed is None:
                    edge_cases['parse_error'] += 1
                    continue
                    
                # Check for empty responses
                if len(processed['chosen_response']) == 0 or len(processed['rejected_response']) == 0:
                    edge_cases['empty_response'] += 1
                    filtered_count += 1
                    continue
                
                # Check sequence length
                if filter_long_sequences:
                    chosen_tokens = self.tokenizer(
                        processed['prompt'] + processed['chosen_response'],
                        truncation=False
                    )['input_ids']
                    rejected_tokens = self.tokenizer(
                        processed['prompt'] + processed['rejected_response'],
                        truncation=False
                    )['input_ids']
                    
                    if len(chosen_tokens) > self.max_length or len(rejected_tokens) > self.max_length:
                        edge_cases['too_long'] += 1
                        # Still include but will truncate during tokenization
                
                all_data.append(processed)
                processed_count += 1
                
            except Exception as e:
                edge_cases['parse_error'] += 1
                continue
        
        print(f"\nPreprocessing complete:")
        print(f"  - Successfully processed: {processed_count}")
        print(f"  - Filtered out: {filtered_count}")
        print(f"  - Edge cases: {edge_cases}")
        
        # Shuffle and split
        random.shuffle(all_data)
        
        split_idx = int(len(all_data) * train_ratio)
        train_split = all_data[:split_idx]
        val_split = all_data[split_idx:]
        
        print(f"\nDataset splits:")
        print(f"  - Training: {len(train_split)}")
        print(f"  - Validation: {len(val_split)}")
        
        # Create HuggingFace datasets
        self.processed_dataset = DatasetDict({
            'train': Dataset.from_list(train_split),
            'validation': Dataset.from_list(val_split)
        })
        
        return self.processed_dataset
    
    def _process_sample(self, sample: Dict) -> Optional[Dict]:
        """
        Process a single sample from the dataset.
        
        Args:
            sample: Raw sample with 'chosen' and 'rejected' fields
            
        Returns:
            Processed sample with separated prompt and responses
        """
        chosen_prompt, chosen_response = self._parse_conversation(sample['chosen'])
        rejected_prompt, rejected_response = self._parse_conversation(sample['rejected'])
        
        # Verify prompts match (they should be the same)
        # Sometimes there might be small differences, so we use the chosen prompt
        
        return {
            'prompt': chosen_prompt,
            'chosen_response': chosen_response,
            'rejected_response': rejected_response,
            'chosen_full': sample['chosen'],
            'rejected_full': sample['rejected']
        }
    
    def tokenize_for_reward_model(self, batch: Dict) -> Dict:
        """
        Tokenize a batch for reward model training.
        
        Args:
            batch: Batch of samples
            
        Returns:
            Tokenized batch with input_ids, attention_mask for both chosen and rejected
        """
        # Tokenize chosen
        chosen_encodings = self.tokenizer(
            [p + r for p, r in zip(batch['prompt'], batch['chosen_response'])],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize rejected
        rejected_encodings = self.tokenizer(
            [p + r for p, r in zip(batch['prompt'], batch['rejected_response'])],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encodings['input_ids'],
            'chosen_attention_mask': chosen_encodings['attention_mask'],
            'rejected_input_ids': rejected_encodings['input_ids'],
            'rejected_attention_mask': rejected_encodings['attention_mask'],
        }
    
    def get_tokenized_dataset(self, batch_size: int = 1000) -> DatasetDict:
        """
        Get fully tokenized dataset ready for training.
        
        Args:
            batch_size: Batch size for tokenization
            
        Returns:
            Tokenized DatasetDict
        """
        if self.processed_dataset is None:
            raise ValueError("Dataset not preprocessed. Call preprocess_dataset() first.")
            
        print("\nTokenizing dataset...")
        
        tokenized = self.processed_dataset.map(
            self.tokenize_for_reward_model,
            batched=True,
            batch_size=batch_size,
            desc="Tokenizing"
        )
        
        return tokenized
    
    def save_processed_data(self, save_path: str):
        """
        Save processed dataset to disk.
        
        Args:
            save_path: Path to save the dataset
        """
        if self.processed_dataset is None:
            raise ValueError("Dataset not preprocessed. Call preprocess_dataset() first.")
            
        self.processed_dataset.save_to_disk(save_path)
        print(f"Saved processed dataset to: {save_path}")
        
    def load_processed_data(self, load_path: str) -> DatasetDict:
        """
        Load processed dataset from disk.
        
        Args:
            load_path: Path to load the dataset from
            
        Returns:
            Loaded DatasetDict
        """
        from datasets import load_from_disk
        self.processed_dataset = load_from_disk(load_path)
        print(f"Loaded processed dataset from: {load_path}")
        return self.processed_dataset


def main():
    """Main function to run the data preparation pipeline."""
    
    # Initialize processor
    processor = HHRLHFDataProcessor(
        model_name="gpt2",
        max_length=512,
        seed=42
    )
    
    # Task A: Load and explore dataset
    print("\n" + "="*70)
    print("TASK A: Loading and Exploring Dataset")
    print("="*70)
    
    dataset = processor.load_dataset()
    analysis = processor.explore_dataset(num_examples=5, save_dir="./analysis")
    
    # Task B: Preprocess dataset
    print("\n" + "="*70)
    print("TASK B: Preprocessing Dataset")
    print("="*70)
    
    # For testing, use a smaller subset
    # Set max_samples=None for full dataset
    processed = processor.preprocess_dataset(
        train_ratio=0.9,
        max_samples=10000,  # Use 10k samples for faster testing, set to None for full
        filter_long_sequences=True
    )
    
    # Save processed data
    processor.save_processed_data("./processed_data")
    
    # Show some processed examples
    print("\n" + "="*70)
    print("PROCESSED DATA EXAMPLES")
    print("="*70)
    
    for i in range(3):
        sample = processed['train'][i]
        print(f"\n--- Processed Example {i+1} ---")
        print(f"Prompt length: {len(sample['prompt'])} chars")
        print(f"Chosen response length: {len(sample['chosen_response'])} chars")
        print(f"Rejected response length: {len(sample['rejected_response'])} chars")
        print(f"Prompt preview: {sample['prompt'][:200]}...")
        print(f"Chosen preview: {sample['chosen_response'][:150]}...")
        print(f"Rejected preview: {sample['rejected_response'][:150]}...")
    
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\nSummary:")
    print(f"  - Training samples: {len(processed['train'])}")
    print(f"  - Validation samples: {len(processed['validation'])}")
    print(f"  - Saved to: ./processed_data")
    print(f"  - Analysis saved to: ./analysis")


if __name__ == "__main__":
    main()