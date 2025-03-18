import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from datasets import load_dataset
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple

class PPOTrainer:
    def __init__(
        self,
        model_path: str,
        ja_features_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        c1: float = 1.0,  # Value loss coefficient
        c2: float = 0.01,  # Entropy coefficient
        batch_size: int = 8,
        max_length: int = 128,
    ):
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Load Japanese features
        with open(ja_features_path, 'r') as f:
            ja_features = json.load(f)['ja']
        self.ja_feature_indices = torch.tensor(ja_features['feature_indices'], device=device)
        
        # Training parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
    
    def compute_reward(self, feature_activations: torch.Tensor) -> torch.Tensor:
        """Compute reward based on Japanese feature activations"""
        # Get activations for Japanese features
        ja_activations = feature_activations[:, self.ja_feature_indices]
        # Reward is the mean activation of Japanese features
        return ja_activations.mean(dim=-1)
    
    def get_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get value estimate from hidden states"""
        # Use the last hidden state to predict value
        value = self.model.get_output_embeddings()(hidden_states[:, -1, :])
        return value.squeeze(-1)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one PPO training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get logits and hidden states
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Get value estimates
        values = self.get_value(hidden_states)
        
        # Compute action log probabilities
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        action_log_probs = torch.gather(
            log_probs, -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        
        # Compute rewards (using Japanese feature activations)
        rewards = self.compute_reward(hidden_states)
        
        # Compute advantages
        advantages = self.compute_gae(
            rewards,
            values,
            torch.cat([values[1:], torch.zeros(1, device=self.device)]),
            torch.zeros_like(rewards),
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy ratio
        ratio = torch.exp(action_log_probs - old_log_probs)
        
        # Compute PPO loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = self.c1 * nn.MSELoss()(values, rewards)
        
        # Compute entropy loss
        entropy_loss = -self.c2 * torch.mean(torch.sum(log_probs * torch.exp(log_probs), dim=-1))
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
        }
    
    def train(
        self,
        dataset_name: str = "Muennighoff/flores200",
        num_epochs: int = 3,
        save_dir: str = "assets/ppo_model",
    ):
        """Train the model using PPO"""
        # Load dataset
        dataset = load_dataset(dataset_name, trust_remote_code=True)
        train_dataset = dataset["train"]
        
        # Create dataloader
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_stats = []
            
            for batch in tqdm(dataloader, desc="Training"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get initial log probabilities
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    logits = outputs.logits
                    log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
                    old_log_probs = torch.gather(
                        log_probs, -1, input_ids[:, 1:].unsqueeze(-1)
                    ).squeeze(-1)
                
                # Training step
                stats = self.train_step(input_ids, attention_mask, old_log_probs)
                epoch_stats.append(stats)
            
            # Print epoch statistics
            avg_stats = {
                k: np.mean([s[k] for s in epoch_stats])
                for k in epoch_stats[0].keys()
            }
            print("\nEpoch Statistics:")
            for k, v in avg_stats.items():
                print(f"{k}: {v:.4f}")
        
        # Save the model
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"\nModel saved to {save_path}")
    
    def collate_fn(self, examples):
        """Collate function for the dataloader"""
        # Tokenize texts
        texts = [ex["ja"] for ex in examples]  # Using Japanese text
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

def main():
    # Initialize trainer
    trainer = PPOTrainer(
        model_path="assets/gpt2-small",
        ja_features_path="assets/filtered_lang_features.json",
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main() 