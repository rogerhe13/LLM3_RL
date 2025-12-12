"""
Part 1.2 Task B: Reward Model Evaluation and Error Analysis
- Report validation accuracy
- Analyze where the model makes mistakes (20+ examples)
"""

import os
import json
import argparse
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_from_disk, load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reward_model import RewardModel, get_device
from train_reward_model import PreferenceDataset, load_and_prepare_data


class RewardModelEvaluator:
    """Evaluator for the Reward Model with detailed error analysis."""
    
    def __init__(
        self,
        model: RewardModel,
        tokenizer,
        device: torch.device,
        output_dir: str = "./evaluation_results"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained reward model
            tokenizer: Tokenizer
            device: Device to use
            output_dir: Directory to save evaluation results
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_full(
        self,
        dataloader: DataLoader,
        collect_errors: bool = True,
        max_errors: int = 50
    ) -> Dict:
        """
        Full evaluation with metrics and error collection.
        
        Args:
            dataloader: DataLoader for evaluation
            collect_errors: Whether to collect error examples
            max_errors: Maximum number of errors to collect
            
        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*60)
        print("REWARD MODEL EVALUATION")
        print("="*60)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        all_chosen_rewards = []
        all_rejected_rewards = []
        all_margins = []
        
        # Error collection
        errors = []
        correct_examples = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                chosen_input_ids = batch['chosen_input_ids'].to(self.device)
                chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
                rejected_input_ids = batch['rejected_input_ids'].to(self.device)
                rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
                
                # Get rewards
                chosen_rewards = self.model(chosen_input_ids, chosen_attention_mask)
                rejected_rewards = self.model(rejected_input_ids, rejected_attention_mask)
                
                # Compute loss
                loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards) + 1e-8).mean()
                
                # Track metrics
                total_loss += loss.item() * chosen_input_ids.size(0)
                
                # Check correctness
                correct_mask = (chosen_rewards > rejected_rewards).squeeze()
                total_correct += correct_mask.sum().item()
                total_samples += chosen_input_ids.size(0)
                
                # Store rewards for analysis
                all_chosen_rewards.extend(chosen_rewards.squeeze().cpu().tolist())
                all_rejected_rewards.extend(rejected_rewards.squeeze().cpu().tolist())
                margins = (chosen_rewards - rejected_rewards).squeeze().cpu().tolist()
                if isinstance(margins, float):
                    margins = [margins]
                all_margins.extend(margins)
                
                # Collect examples
                if collect_errors:
                    for i in range(len(chosen_rewards)):
                        is_correct = correct_mask[i].item() if correct_mask.dim() > 0 else correct_mask.item()
                        
                        example = {
                            'chosen_text': batch['chosen_text'][i][:1000],  # Truncate for storage
                            'rejected_text': batch['rejected_text'][i][:1000],
                            'chosen_reward': chosen_rewards[i].item(),
                            'rejected_reward': rejected_rewards[i].item(),
                            'margin': (chosen_rewards[i] - rejected_rewards[i]).item(),
                            'is_correct': is_correct
                        }
                        
                        if not is_correct and len(errors) < max_errors:
                            errors.append(example)
                        elif is_correct and len(correct_examples) < 10:
                            correct_examples.append(example)
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        results = {
            'validation_loss': avg_loss,
            'validation_accuracy': accuracy,
            'total_samples': total_samples,
            'correct_predictions': total_correct,
            'incorrect_predictions': total_samples - total_correct,
            'avg_chosen_reward': np.mean(all_chosen_rewards),
            'avg_rejected_reward': np.mean(all_rejected_rewards),
            'avg_margin': np.mean(all_margins),
            'std_margin': np.std(all_margins),
            'reward_statistics': {
                'chosen_mean': float(np.mean(all_chosen_rewards)),
                'chosen_std': float(np.std(all_chosen_rewards)),
                'chosen_min': float(np.min(all_chosen_rewards)),
                'chosen_max': float(np.max(all_chosen_rewards)),
                'rejected_mean': float(np.mean(all_rejected_rewards)),
                'rejected_std': float(np.std(all_rejected_rewards)),
                'rejected_min': float(np.min(all_rejected_rewards)),
                'rejected_max': float(np.max(all_rejected_rewards)),
            }
        }
        
        # Print results
        print(f"\n{'='*40}")
        print("EVALUATION RESULTS")
        print(f"{'='*40}")
        print(f"Validation Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
        print(f"\nReward Statistics:")
        print(f"  Chosen rewards - Mean: {np.mean(all_chosen_rewards):.4f}, Std: {np.std(all_chosen_rewards):.4f}")
        print(f"  Rejected rewards - Mean: {np.mean(all_rejected_rewards):.4f}, Std: {np.std(all_rejected_rewards):.4f}")
        print(f"  Margin (chosen - rejected) - Mean: {np.mean(all_margins):.4f}, Std: {np.std(all_margins):.4f}")
        
        # Store for visualization
        self.all_chosen_rewards = all_chosen_rewards
        self.all_rejected_rewards = all_rejected_rewards
        self.all_margins = all_margins
        self.errors = errors
        self.correct_examples = correct_examples
        
        return results
    
    def analyze_errors(self, num_examples: int = 20) -> List[Dict]:
        """
        Analyze error examples in detail.
        
        Args:
            num_examples: Number of error examples to analyze
            
        Returns:
            List of analyzed error examples
        """
        if not hasattr(self, 'errors') or not self.errors:
            print("No errors collected. Run evaluate_full first.")
            return []
        
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS ({min(num_examples, len(self.errors))} examples)")
        print(f"{'='*60}")
        
        analyzed_errors = []
        
        for i, error in enumerate(self.errors[:num_examples]):
            print(f"\n{'─'*60}")
            print(f"ERROR EXAMPLE {i+1}")
            print(f"{'─'*60}")
            
            # Parse chosen and rejected to extract just the last response
            chosen_prompt, chosen_response = self._extract_last_response(error['chosen_text'])
            _, rejected_response = self._extract_last_response(error['rejected_text'])
            
            # Analyze characteristics
            analysis = {
                'example_id': i + 1,
                'chosen_reward': error['chosen_reward'],
                'rejected_reward': error['rejected_reward'],
                'margin': error['margin'],
                'chosen_response_length': len(chosen_response),
                'rejected_response_length': len(rejected_response),
                'length_diff': len(chosen_response) - len(rejected_response),
                'chosen_has_more_words': len(chosen_response.split()) > len(rejected_response.split()),
            }
            
            # Try to identify error patterns
            error_pattern = self._identify_error_pattern(
                chosen_response, 
                rejected_response,
                error['chosen_reward'],
                error['rejected_reward']
            )
            analysis['likely_error_pattern'] = error_pattern
            
            print(f"Prompt (truncated):\n{chosen_prompt[:300]}...")
            print(f"\n[CHOSEN - Should be preferred] (Reward: {error['chosen_reward']:.4f})")
            print(f"{chosen_response[:400]}{'...' if len(chosen_response) > 400 else ''}")
            print(f"\n[REJECTED - Model preferred this] (Reward: {error['rejected_reward']:.4f})")
            print(f"{rejected_response[:400]}{'...' if len(rejected_response) > 400 else ''}")
            print(f"\nAnalysis:")
            print(f"  - Chosen response length: {analysis['chosen_response_length']} chars")
            print(f"  - Rejected response length: {analysis['rejected_response_length']} chars")
            print(f"  - Margin: {analysis['margin']:.4f}")
            print(f"  - Likely error pattern: {error_pattern}")
            
            analysis['prompt'] = chosen_prompt[:500]
            analysis['chosen_response'] = chosen_response[:500]
            analysis['rejected_response'] = rejected_response[:500]
            
            analyzed_errors.append(analysis)
        
        return analyzed_errors
    
    def _extract_last_response(self, text: str) -> Tuple[str, str]:
        """Extract the prompt and last assistant response from conversation."""
        parts = text.split("Assistant:")
        if len(parts) < 2:
            return text, ""
        
        response = parts[-1].strip()
        prompt = "Assistant:".join(parts[:-1]).strip()
        
        return prompt, response
    
    def _identify_error_pattern(
        self,
        chosen_response: str,
        rejected_response: str,
        chosen_reward: float,
        rejected_reward: float
    ) -> str:
        """Try to identify common error patterns."""
        patterns = []
        
        chosen_len = len(chosen_response)
        rejected_len = len(rejected_response)
        
        # Length-based patterns
        if rejected_len > chosen_len * 1.5:
            patterns.append("Length bias (longer rejected)")
        elif chosen_len > rejected_len * 1.5:
            patterns.append("Model missed quality despite length")
        
        # Very short responses
        if chosen_len < 50:
            patterns.append("Very short chosen response")
        if rejected_len < 50:
            patterns.append("Very short rejected response")
        
        # Check for refusal patterns in chosen
        refusal_keywords = ["i can't", "i cannot", "i'm not able", "sorry", "i won't", "i don't think"]
        chosen_lower = chosen_response.lower()
        rejected_lower = rejected_response.lower()
        
        if any(kw in chosen_lower for kw in refusal_keywords):
            if not any(kw in rejected_lower for kw in refusal_keywords):
                patterns.append("Chosen is refusal, rejected is engagement")
        
        # Check for harmful content indicators
        harmful_keywords = ["here's how", "step 1", "first,", "to do this"]
        if any(kw in rejected_lower for kw in harmful_keywords):
            if any(kw in chosen_lower for kw in refusal_keywords):
                patterns.append("Harmful content preference")
        
        # Very close margin
        if abs(chosen_reward - rejected_reward) < 0.1:
            patterns.append("Very close rewards (ambiguous)")
        
        if not patterns:
            patterns.append("Unclear pattern")
        
        return "; ".join(patterns)
    
    def generate_error_summary(self) -> Dict:
        """Generate a summary of error patterns."""
        if not hasattr(self, 'errors') or not self.errors:
            return {}
        
        print(f"\n{'='*60}")
        print("ERROR PATTERN SUMMARY")
        print(f"{'='*60}")
        
        # Analyze patterns across all errors
        pattern_counts = {}
        length_diffs = []
        margins = []
        
        for error in self.errors:
            chosen_prompt, chosen_response = self._extract_last_response(error['chosen_text'])
            _, rejected_response = self._extract_last_response(error['rejected_text'])
            
            pattern = self._identify_error_pattern(
                chosen_response, rejected_response,
                error['chosen_reward'], error['rejected_reward']
            )
            
            for p in pattern.split("; "):
                pattern_counts[p] = pattern_counts.get(p, 0) + 1
            
            length_diffs.append(len(rejected_response) - len(chosen_response))
            margins.append(error['margin'])
        
        summary = {
            'total_errors': len(self.errors),
            'pattern_distribution': pattern_counts,
            'avg_length_diff_in_errors': np.mean(length_diffs),
            'avg_margin_in_errors': np.mean(margins),
            'std_margin_in_errors': np.std(margins)
        }
        
        print(f"\nTotal errors analyzed: {len(self.errors)}")
        print(f"\nError pattern distribution:")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
            print(f"  - {pattern}: {count} ({100*count/len(self.errors):.1f}%)")
        
        print(f"\nIn error cases:")
        print(f"  - Avg length diff (rejected - chosen): {np.mean(length_diffs):.1f} chars")
        print(f"  - Avg margin: {np.mean(margins):.4f}")
        
        return summary
    
    def plot_reward_distributions(self):
        """Plot reward distributions and save."""
        if not hasattr(self, 'all_chosen_rewards'):
            print("No reward data available. Run evaluate_full first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Reward distributions
        axes[0, 0].hist(self.all_chosen_rewards, bins=50, alpha=0.5, label='Chosen', color='green')
        axes[0, 0].hist(self.all_rejected_rewards, bins=50, alpha=0.5, label='Rejected', color='red')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Margin distribution
        axes[0, 1].hist(self.all_margins, bins=50, color='blue', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', label='Zero margin')
        axes[0, 1].axvline(x=np.mean(self.all_margins), color='green', linestyle='--', 
                          label=f'Mean: {np.mean(self.all_margins):.3f}')
        axes[0, 1].set_xlabel('Margin (Chosen - Rejected)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Reward Margins')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter plot of chosen vs rejected rewards
        axes[1, 0].scatter(self.all_rejected_rewards, self.all_chosen_rewards, alpha=0.3, s=10)
        min_val = min(min(self.all_chosen_rewards), min(self.all_rejected_rewards))
        max_val = max(max(self.all_chosen_rewards), max(self.all_rejected_rewards))
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (equal rewards)')
        axes[1, 0].set_xlabel('Rejected Reward')
        axes[1, 0].set_ylabel('Chosen Reward')
        axes[1, 0].set_title('Chosen vs Rejected Rewards')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Error margins vs correct margins
        correct_margins = [m for m, c, r in zip(self.all_margins, self.all_chosen_rewards, self.all_rejected_rewards) if c > r]
        error_margins = [m for m, c, r in zip(self.all_margins, self.all_chosen_rewards, self.all_rejected_rewards) if c <= r]
        
        if correct_margins:
            axes[1, 1].hist(correct_margins, bins=30, alpha=0.5, label=f'Correct ({len(correct_margins)})', color='green')
        if error_margins:
            axes[1, 1].hist(error_margins, bins=30, alpha=0.5, label=f'Errors ({len(error_margins)})', color='red')
        axes[1, 1].set_xlabel('Margin')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Margin Distribution: Correct vs Errors')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "reward_distributions.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"\nReward distribution plots saved to: {plot_path}")
    
    def save_evaluation_report(self, results: Dict, analyzed_errors: List[Dict], error_summary: Dict):
        """Save complete evaluation report."""
        report = {
            'evaluation_results': results,
            'error_summary': error_summary,
            'detailed_errors': analyzed_errors[:25],  # Save top 25 errors
            'correct_examples': self.correct_examples if hasattr(self, 'correct_examples') else []
        }
        
        # Save JSON report
        report_path = os.path.join(self.output_dir, "evaluation_report.json")
        
        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        report = convert_numpy(report)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nEvaluation report saved to: {report_path}")
        
        # Save markdown report for easy reading
        md_report = self._generate_markdown_report(results, analyzed_errors, error_summary)
        md_path = os.path.join(self.output_dir, "evaluation_report.md")
        with open(md_path, 'w') as f:
            f.write(md_report)
        print(f"Markdown report saved to: {md_path}")
    
    def _generate_markdown_report(self, results: Dict, analyzed_errors: List[Dict], error_summary: Dict) -> str:
        """Generate a markdown format evaluation report."""
        report = f"""# Reward Model Evaluation Report

## Summary

| Metric | Value |
|--------|-------|
| Validation Accuracy | {results['validation_accuracy']:.4f} ({results['validation_accuracy']*100:.2f}%) |
| Validation Loss | {results['validation_loss']:.4f} |
| Total Samples | {results['total_samples']} |
| Correct Predictions | {results['correct_predictions']} |
| Incorrect Predictions | {results['incorrect_predictions']} |

## Reward Statistics

| Metric | Chosen | Rejected |
|--------|--------|----------|
| Mean | {results['reward_statistics']['chosen_mean']:.4f} | {results['reward_statistics']['rejected_mean']:.4f} |
| Std | {results['reward_statistics']['chosen_std']:.4f} | {results['reward_statistics']['rejected_std']:.4f} |
| Min | {results['reward_statistics']['chosen_min']:.4f} | {results['reward_statistics']['rejected_min']:.4f} |
| Max | {results['reward_statistics']['chosen_max']:.4f} | {results['reward_statistics']['rejected_max']:.4f} |

## Error Analysis Summary

Total errors analyzed: {error_summary.get('total_errors', 0)}

### Error Pattern Distribution

| Pattern | Count | Percentage |
|---------|-------|------------|
"""
        
        for pattern, count in sorted(error_summary.get('pattern_distribution', {}).items(), key=lambda x: -x[1]):
            pct = 100 * count / error_summary.get('total_errors', 1)
            report += f"| {pattern} | {count} | {pct:.1f}% |\n"
        
        report += f"""

### Key Findings

- Average length difference in errors (rejected - chosen): {error_summary.get('avg_length_diff_in_errors', 0):.1f} chars
- Average margin in errors: {error_summary.get('avg_margin_in_errors', 0):.4f}

## Detailed Error Examples

"""
        
        for error in analyzed_errors[:10]:
            report += f"""
### Error Example {error['example_id']}

**Prompt (truncated):**
```
{error.get('prompt', 'N/A')[:300]}...
```

**Chosen Response** (Reward: {error['chosen_reward']:.4f})
```
{error.get('chosen_response', 'N/A')[:300]}...
```

**Rejected Response** (Reward: {error['rejected_reward']:.4f})
```
{error.get('rejected_response', 'N/A')[:300]}...
```

**Analysis:**
- Margin: {error['margin']:.4f}
- Chosen length: {error['chosen_response_length']} chars
- Rejected length: {error['rejected_response_length']} chars
- Likely pattern: {error['likely_error_pattern']}

---
"""
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Reward Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Backbone model name")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=5000, help="Maximum samples for evaluation")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--num_error_examples", type=int, default=25, help="Number of error examples to analyze")
    parser.add_argument("--processed_data_path", type=str, default=None, help="Path to preprocessed data")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("REWARD MODEL EVALUATION - Part 1.2 Task B")
    print("="*60)
    
    # Get device
    device = get_device()
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"\nLoading model from: {args.model_path}")
    model = RewardModel.load_pretrained(args.model_path, device=device)
    
    # Load validation data
    print("\nLoading validation data...")
    _, val_data = load_and_prepare_data(
        processed_data_path=args.processed_data_path,
        max_samples=args.max_samples
    )
    
    # Create dataset and dataloader - optimize for GPU
    val_dataset = PreferenceDataset(val_data, tokenizer, args.max_length)
    use_cuda = device.type == 'cuda'
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda
    )
    
    # Initialize evaluator
    evaluator = RewardModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = evaluator.evaluate_full(val_loader, collect_errors=True, max_errors=50)
    
    # Analyze errors
    analyzed_errors = evaluator.analyze_errors(num_examples=args.num_error_examples)
    
    # Generate error summary
    error_summary = evaluator.generate_error_summary()
    
    # Plot distributions
    evaluator.plot_reward_distributions()
    
    # Save complete report
    evaluator.save_evaluation_report(results, analyzed_errors, error_summary)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()