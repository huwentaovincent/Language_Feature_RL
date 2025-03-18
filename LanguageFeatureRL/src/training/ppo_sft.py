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
from sae_lens import SAE, HookedSAETransformer

class PPOTrainer:
    def __init__(
        self,
        model_path: str,
        features_path: str,
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
        # We need both the HookedSAETransformer for feature extraction and GPT2LMHeadModel for training
        self.sae_model = HookedSAETransformer.from_pretrained(model_path, device=device)
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Load SAE for feature extraction
        self.sae, _, _ = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.7.hook_resid_pre",
            device=device,
        )
        
        # Create a value head on top of the model
        self.value_head = nn.Linear(self.model.config.n_embd, 1).to(device)
        self.optimizer = AdamW(list(self.model.parameters()) + list(self.value_head.parameters()), 
                              lr=learning_rate)
        
        # Load Japanese features
        with open(features_path, 'r') as f:
            features = json.load(f)
            if 'ja' in features:
                ja_features = features['ja']
            else:
                # Handle the case where data is not nested by language
                ja_features = features
        
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
    
    def compute_feature_activations(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute SAE feature activations for the given input"""
        # Use HookedSAETransformer to get activations
        _, cache = self.sae_model.run_with_cache_with_saes(
            input_ids=input_ids,
            attention_mask=attention_mask,
            saes=[self.sae]
        )
        
        # Get the SAE feature activations 
        feature_acts = cache["blocks.7.hook_resid_pre.hook_sae_acts_post"]
        return feature_acts
    
    def compute_reward(self, feature_activations: torch.Tensor) -> torch.Tensor:
        """Compute reward based on Japanese feature activations"""
        # Extract only Japanese features
        ja_activations = torch.zeros(
            (feature_activations.shape[0], feature_activations.shape[1], len(self.ja_feature_indices)),
            device=self.device
        )
        
        for i, idx in enumerate(self.ja_feature_indices):
            ja_activations[:, :, i] = feature_activations[:, :, idx]
        
        # Reward is mean activation across Japanese features, summed across sequence
        sequence_rewards = ja_activations.mean(dim=2)  # Average across features
        
        # We return the sum across the sequence, since we want to reward the entire generation
        return sequence_rewards.sum(dim=1)
    
    def get_value(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get value estimate from hidden states"""
        # Use a dedicated value head for better estimates
        # We use the mean of all token representations
        mean_hidden = hidden_states.mean(dim=1)
        value = self.value_head(mean_hidden)
        return value.squeeze(-1)
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute advantages for PPO using simplified approach for language models"""
        # For language models, we can simplify by using the reward-value difference
        advantages = rewards - values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages
    
    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one PPO training step"""
        self.model.train()
        self.value_head.train()
        
        # STAGE 1: Reference model forward pass (to get old probs and features)
        with torch.no_grad():
            # Get feature activations for reward computation
            feature_activations = self.compute_feature_activations(input_ids, attention_mask)
            rewards = self.compute_reward(feature_activations)
            
            # Get old log probs
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            old_logits = outputs.logits
            old_hidden = outputs.hidden_states[-1]
            
            old_log_probs = torch.log_softmax(old_logits[:, :-1], dim=-1)
            old_action_log_probs = torch.gather(
                old_log_probs, -1, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            
            # Calculate mask for padding tokens
            valid_tokens_mask = attention_mask[:, 1:] > 0
            old_action_log_probs = old_action_log_probs * valid_tokens_mask
            
            # Get old values
            old_values = self.get_value(old_hidden)
        
        # STAGE 2: Policy model forward pass (updated model)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]
        
        # Get new log probs
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        action_log_probs = torch.gather(
            log_probs, -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        action_log_probs = action_log_probs * valid_tokens_mask
        
        # Get new values
        values = self.get_value(hidden_states)
        
        # Compute advantages
        advantages = self.compute_advantages(rewards, old_values)
        
        # Compute policy ratio
        # Sum log probs across the sequence to get per-sequence log probs
        old_sequence_log_probs = old_action_log_probs.sum(dim=1)
        sequence_log_probs = action_log_probs.sum(dim=1)
        ratio = torch.exp(sequence_log_probs - old_sequence_log_probs)
        
        # Clipped policy loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = self.c1 * nn.MSELoss()(values, rewards)
        
        # Entropy loss (at token level, averaged across sequence)
        entropy_per_token = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
        masked_entropy = entropy_per_token * valid_tokens_mask
        entropy = masked_entropy.sum(dim=1) / valid_tokens_mask.sum(dim=1)
        entropy_loss = -self.c2 * entropy.mean()
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 0.5)
        self.optimizer.step()
        
        # Return statistics
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
            "mean_value": values.mean().item(),
            "mean_ratio": ratio.mean().item(),
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
        
        # Create a subset that only has Japanese text
        ja_texts = []
        for split in ["train", "dev"]:
            if split in dataset:
                for item in dataset[split]:
                    if "ja" in item:
                        ja_texts.append({"text": item["ja"]})
        
        if not ja_texts:
            raise ValueError("No Japanese texts found in the dataset")
        
        print(f"Loaded {len(ja_texts)} Japanese texts for training")
        
        # Create dataloader
        dataloader = DataLoader(
            ja_texts,
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
                
                # Training step
                stats = self.train_step(input_ids, attention_mask)
                epoch_stats.append(stats)
                
                # Print occasional stats
                if len(epoch_stats) % 10 == 0:
                    avg_stats = {
                        k: np.mean([s[k] for s in epoch_stats[-10:]])
                        for k in epoch_stats[-1].keys()
                    }
                    stat_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_stats.items()])
                    print(f"Step {len(epoch_stats)}: {stat_str}")
            
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
        
        # Also save the value head
        torch.save(self.value_head.state_dict(), save_path / "value_head.pt")
        
        print(f"\nModel saved to {save_path}")
    
    def collate_fn(self, examples):
        """Collate function for the dataloader"""
        # Tokenize texts
        texts = [ex["text"] for ex in examples]
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
        features_path="assets/filtered_lang_features.json",
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main()