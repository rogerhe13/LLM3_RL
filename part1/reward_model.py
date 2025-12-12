"""
Part 1.2: Reward Model Implementation
Reward model for RLHF using GPT-2 as backbone.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RewardModel(nn.Module):
    """
    Reward Model for RLHF.
    
    Architecture:
        GPT-2 (backbone) → Last token hidden state → Linear layer → Scalar reward
    
    The model learns to assign higher rewards to preferred responses.
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        dropout: float = 0.1
    ):
        """
        Initialize the reward model.
        
        Args:
            model_name: Name of the pretrained model to use as backbone
            dropout: Dropout rate for the reward head
        """
        super().__init__()
        
        self.model_name = model_name
        
        # Load pretrained model as backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        
        # Get hidden size from config
        hidden_size = self.config.hidden_size  # 768 for gpt2
        
        # Reward head: projects hidden state to scalar reward
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize reward head weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the reward head weights."""
        for module in self.reward_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to compute rewards.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            rewards: Scalar rewards [batch_size, 1]
        """
        # Get hidden states from backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the last hidden state
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Get the hidden state of the last non-padded token
        # We use attention_mask to find the last real token
        batch_size = input_ids.shape[0]
        
        # Find the index of the last non-padded token for each sequence
        # attention_mask: 1 for real tokens, 0 for padding
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        sequence_lengths = sequence_lengths.clamp(min=0)  # Safety clamp
        
        # Gather the hidden state at the last token position
        last_hidden = hidden_states[
            torch.arange(batch_size, device=hidden_states.device),
            sequence_lengths
        ]  # [batch_size, hidden_size]
        
        # Compute reward
        rewards = self.reward_head(last_hidden)  # [batch_size, 1]
        
        return rewards
    
    def compute_pairwise_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> tuple:
        """
        Compute the pairwise ranking loss.
        
        Loss = -log(σ(r(x, y_chosen) - r(x, y_rejected)))
        
        Args:
            chosen_input_ids: Token IDs for chosen responses
            chosen_attention_mask: Attention mask for chosen responses
            rejected_input_ids: Token IDs for rejected responses
            rejected_attention_mask: Attention mask for rejected responses
            
        Returns:
            loss: Scalar loss value
            chosen_rewards: Rewards for chosen responses
            rejected_rewards: Rewards for rejected responses
        """
        # Compute rewards for both chosen and rejected
        chosen_rewards = self.forward(chosen_input_ids, chosen_attention_mask)
        rejected_rewards = self.forward(rejected_input_ids, rejected_attention_mask)
        
        # Compute pairwise ranking loss
        # L = -log(σ(r_chosen - r_rejected))
        # This is equivalent to binary cross entropy with logits
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards) + 1e-8).mean()
        
        return loss, chosen_rewards, rejected_rewards
    
    def get_reward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Get reward for a single input (inference mode).
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            reward: Scalar reward value
        """
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)
    
    def save_pretrained(self, save_path: str):
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'config': self.config
        }, save_path)
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_pretrained(cls, load_path: str, device: torch.device = None):
        """Load model from disk."""
        checkpoint = torch.load(load_path, map_location=device)
        model = cls(model_name=checkpoint['model_name'])
        model.load_state_dict(checkpoint['model_state_dict'])
        if device:
            model = model.to(device)
        print(f"Model loaded from {load_path}")
        return model


def get_device():
    """Get the best available device (CUDA for NVIDIA, MPS for Mac, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


if __name__ == "__main__":
    # Quick test
    print("Testing Reward Model...")
    
    device = get_device()
    
    # Initialize model
    model = RewardModel(model_name="gpt2")
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    test_texts = [
        "Human: Hello!\n\nAssistant: Hi there! How can I help you?",
        "Human: Hello!\n\nAssistant: Go away."
    ]
    
    inputs = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    rewards = model(inputs['input_ids'], inputs['attention_mask'])
    print(f"Test rewards: {rewards.squeeze().tolist()}")
    
    print("\nReward Model test passed!")