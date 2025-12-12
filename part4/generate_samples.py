"""
Part 4: Generate samples from all models for evaluation.

This script generates responses from:
- Base GPT-2
- PPO-trained model
- GRPO-trained model  
- DPO-trained model

Outputs saved for GPT-4-as-judge evaluation.
"""

import os
import json
import argparse
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.append('../part2')
from policy_model import PolicyModel, get_device


def load_test_prompts(num_prompts: int = 100) -> List[str]:
    """Load test prompts from HH-RLHF dataset."""
    dataset = load_dataset("Anthropic/hh-rlhf", split="test")
    
    prompts = []
    for item in dataset:
        chosen = item['chosen']
        if '\n\nHuman:' in chosen and '\n\nAssistant:' in chosen:
            prompt = chosen.split('\n\nAssistant:')[0] + '\n\nAssistant:'
            prompts.append(prompt)
        
        if len(prompts) >= num_prompts:
            break
    
    return prompts


def generate_response(model, tokenizer, prompt: str, device, max_new_tokens: int = 128) -> str:
    """Generate a single response."""
    encoding = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=512 - max_new_tokens
    ).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'generate'):
            # PolicyModel
            output_ids = model.generate(
                encoding['input_ids'],
                encoding['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        else:
            # HuggingFace model
            output_ids = model.generate(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
    
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(encoding['input_ids'][0], skip_special_tokens=True)
    response = full_text[len(prompt_text):]
    
    return response.strip()


def load_model(model_path: str, device, is_base: bool = False):
    """Load a model from path."""
    if is_base:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    else:
        model = PolicyModel.load_pretrained(model_path, device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Generate samples from all models")
    parser.add_argument("--base_model", type=str, default="gpt2")
    parser.add_argument("--ppo_model", type=str, default="../part2/ppo_output/final")
    parser.add_argument("--grpo_model", type=str, default="../part2/grpo_output/final")
    parser.add_argument("--dpo_model", type=str, default="../part3/dpo_output/final")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./samples")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Load test prompts
    print(f"Loading {args.num_samples} test prompts...")
    prompts = load_test_prompts(args.num_samples)
    print(f"Loaded {len(prompts)} prompts")
    
    # Models to evaluate
    models_config = {
        'base': {'path': args.base_model, 'is_base': True},
        'ppo': {'path': args.ppo_model, 'is_base': False},
        'grpo': {'path': args.grpo_model, 'is_base': False},
        'dpo': {'path': args.dpo_model, 'is_base': False}
    }
    
    all_samples = []
    
    for model_name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"Generating samples from {model_name.upper()} model...")
        print(f"{'='*60}")
        
        try:
            model = load_model(config['path'], device, config['is_base'])
            
            responses = []
            for i, prompt in enumerate(tqdm(prompts, desc=f"{model_name} generation")):
                response = generate_response(model, tokenizer, prompt, device, args.max_new_tokens)
                responses.append(response)
                
                # Store sample
                if i < len(all_samples):
                    all_samples[i][f'{model_name}_response'] = response
                else:
                    all_samples.append({
                        'id': i,
                        'prompt': prompt,
                        f'{model_name}_response': response
                    })
            
            # Save individual model samples
            model_samples = [{'id': i, 'prompt': prompts[i], 'response': responses[i]} 
                           for i in range(len(prompts))]
            
            with open(os.path.join(args.output_dir, f'{model_name}_samples.json'), 'w') as f:
                json.dump(model_samples, f, indent=2)
            
            print(f"Saved {len(responses)} samples for {model_name}")
            
            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            print("Skipping this model...")
            continue
    
    # Save combined samples
    with open(os.path.join(args.output_dir, 'all_samples.json'), 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SAMPLE GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Samples saved to {args.output_dir}/")
    print(f"  - all_samples.json (combined)")
    print(f"  - base_samples.json")
    print(f"  - ppo_samples.json")
    print(f"  - grpo_samples.json")
    print(f"  - dpo_samples.json")


if __name__ == "__main__":
    main()
