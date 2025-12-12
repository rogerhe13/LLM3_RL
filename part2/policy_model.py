"""
Part 2: Policy Model for RLHF
GPT-2 based policy model for PPO and GRPO training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Tuple, Dict
import copy


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


class PolicyModel(nn.Module):
    """
    Policy Model for RLHF.
    
    This is the model we want to optimize to generate better responses.
    Uses GPT-2 as the base model.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.config = self.model.config
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss (optional)
            
        Returns:
            Dictionary with logits and optionally loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return {
            'logits': outputs.logits,
            'loss': outputs.loss if labels is not None else None
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate responses.
        
        Args:
            input_ids: Prompt token IDs
            attention_mask: Attention mask
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling
            pad_token_id: Pad token ID
            
        Returns:
            Generated token IDs (including prompt)
        """
        # Ensure temperature is not too low to avoid numerical issues
        temperature = max(temperature, 0.1)
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=50,  # Add top_k for stability
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            repetition_penalty=1.1,  # Prevent repetition
        )
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for the response tokens.
        
        Args:
            input_ids: Full sequence (prompt + response) [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            response_mask: Mask indicating response tokens [batch_size, seq_len]
            
        Returns:
            Log probabilities for response tokens [batch_size]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = response_mask[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, seq_len-1]
        
        # Mask and sum log probs for response tokens only
        masked_log_probs = token_log_probs * shift_mask.float()
        sequence_log_probs = masked_log_probs.sum(dim=-1)  # [batch_size]
        
        # Normalize by response length
        response_lengths = shift_mask.sum(dim=-1).clamp(min=1)
        normalized_log_probs = sequence_log_probs / response_lengths
        
        return sequence_log_probs, normalized_log_probs
    
    def get_per_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get per-token log probabilities.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Per-token log probs [batch_size, seq_len-1]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    
    def save_pretrained(self, save_path: str):
        """Save model."""
        self.model.save_pretrained(save_path)
        
    @classmethod
    def load_pretrained(cls, load_path: str):
        """Load model."""
        model = cls.__new__(cls)
        super(PolicyModel, model).__init__()
        model.model = AutoModelForCausalLM.from_pretrained(load_path)
        model.config = model.model.config
        model.model_name = load_path
        return model


class ReferenceModel:
    """
    Reference model wrapper (frozen, for KL computation).
    """
    
    def __init__(self, model: PolicyModel):
        """Create a frozen copy of the policy model."""
        self.model = copy.deepcopy(model)
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def get_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get log probs from reference model (no gradients)."""
        with torch.no_grad():
            return self.model.get_log_probs(input_ids, attention_mask, response_mask)
    
    def get_per_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get per-token log probs from reference model."""
        with torch.no_grad():
            return self.model.get_per_token_log_probs(input_ids, attention_mask)


def compute_kl_divergence(
    policy_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference.
    
    KL(policy || reference) = policy_log_prob - reference_log_prob
    
    Args:
        policy_log_probs: Per-token log probs from policy [batch_size, seq_len]
        reference_log_probs: Per-token log probs from reference [batch_size, seq_len]
        response_mask: Mask for response tokens [batch_size, seq_len]
        
    Returns:
        KL divergence per sequence [batch_size]
    """
    # KL divergence per token
    kl_per_token = policy_log_probs - reference_log_probs
    
    # Mask to response tokens only
    masked_kl = kl_per_token * response_mask[:, 1:].float()
    
    # Sum over sequence
    kl_sum = masked_kl.sum(dim=-1)
    
    return kl_sum


if __name__ == "__main__":
    # Quick test
    print("Testing Policy Model...")
    
    device = get_device()
    
    # Initialize
    policy = PolicyModel("gpt2").to(device)
    reference = ReferenceModel(policy)
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test generation
    prompt = "Human: What is the capital of France?\n\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    outputs = policy.generate(
        inputs['input_ids'],
        inputs['attention_mask'],
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated}")
    
    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("\nPolicy Model test passed! ✓")