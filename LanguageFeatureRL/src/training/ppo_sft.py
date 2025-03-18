import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from datasets import load_dataset
from pathlib import Path
import json
import os
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import wandb
import logging
from datetime import datetime

from training.training_utils import (
    TrainingTracker, 
    set_seed, 
    JapaneseTextDataset, 
    get_dataloader
)

from sae_lens import SAE, HookedSAETransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ppo_trainer")

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
        output_dir: str = "assets/ppo_model",
        wandb_project: Optional[str] = "language-feature-rl",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        use_wandb: bool = True,
        seed: int = 42,
    ):
        """
        Initialize the PPO Trainer.
        
        Args:
            model_path: Path to the pretrained model
            features_path: Path to the filtered language features
            device: Device to use for training
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for rewards
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy bonus coefficient
            batch_size: Training batch size
            max_length: Maximum sequence length
            output_dir: Directory to save model and logs
            wandb_project: W&B project name
            wandb_entity: W&B entity (username or team name)
            wandb_run_name: W&B run name (if None, auto-generated)
            use_wandb: Whether to use W&B for tracking
            seed: Random seed for reproducibility
        """
        # Set random seed
        set_seed(seed)
        
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the HookedSAETransformer for feature activation
        logger.info(f"Initializing HookedSAETransformer from {model_path}...")
        self.sae_model = HookedSAETransformer.from_pretrained(model_path, device=device)
        
        # Initialize the GPT2LMHeadModel for language modeling and training
        logger.info(f"Initializing GPT2LMHeadModel from {model_path}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        
        # Load SAE for feature extraction
        logger.info("Loading SAE for feature extraction...")
        self.sae, _, _ = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.7.hook_resid_pre",
            device=device,
        )
        
        # Cache the hook name for feature activation
        self.sae_hook_name = self.sae.cfg.hook_name
        self.hook_activation_name = f"{self.sae_hook_name}.hook_sae_acts_post"
        
        # Create a value head on top of the model
        self.value_head = nn.Linear(self.model.config.n_embd, 1).to(device)
        self.optimizer = AdamW(
            list(self.model.parameters()) + list(self.value_head.parameters()), 
            lr=learning_rate
        )
        
        # Load Japanese features
        logger.info(f"Loading Japanese features from {features_path}...")
        with open(features_path, 'r') as f:
            features_data = json.load(f)
            
            # Handle different possible JSON structures
            if 'languages' in features_data and 'ja' in features_data['languages']:
                ja_features = features_data['languages']['ja']
            elif 'ja' in features_data:
                ja_features = features_data['ja']
            else:
                # Assume it's flat and contains feature_indices directly
                ja_features = features_data
        
        self.ja_feature_indices = torch.tensor(ja_features['feature_indices'], device=device)
        logger.info(f"Loaded {len(self.ja_feature_indices)} Japanese features")
        
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
        
        # Initialize tracker
        self.tracker = TrainingTracker(
            output_dir=output_dir,
            model_name="gpt2_japanese_ppo",
            save_every=100,
            log_every=10
        )
        
        # Initialize W&B if requested
        self.use_wandb = use_wandb
        if use_wandb:
            # Create a unique run name if not provided
            if wandb_run_name is None:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                wandb_run_name = f"ppo_japanese_{timestamp}"
            
            # Initialize W&B
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_run_name,
                config={
                    "model_path": model_path,
                    "features_path": features_path,
                    "learning_rate": learning_rate,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda,
                    "clip_epsilon": clip_epsilon,
                    "c1": c1,
                    "c2": c2,
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "seed": seed,
                    "num_japanese_features": len(self.ja_feature_indices),
                }
            )
            
            # Log model architecture
            wandb.watch(self.model, log="all", log_freq=10)
            wandb.watch(self.value_head, log="all", log_freq=10)
        
        logger.info("PPO Trainer initialized successfully")
    
    def get_sae_feature_activations(self, input_ids, attention_mask=None):
        """
        Get SAE feature activations for the input text.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Feature activations tensor [batch_size, seq_len, n_features]
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Run the model with SAE to get feature activations
        with torch.no_grad():
            _, cache = self.sae_model.run_with_cache_with_saes(
                input_ids=input_ids,
                attention_mask=attention_mask,
                saes=[self.sae]
            )
            
            # Extract feature activations from cache
            if self.hook_activation_name in cache:
                feature_activations = cache[self.hook_activation_name]
            else:
                # Log all cache keys if the expected key is not found
                logger.error(f"Hook activation name {self.hook_activation_name} not found in cache")
                logger.error(f"Available cache keys: {list(cache.keys())}")
                raise KeyError(f"Hook activation name {self.hook_activation_name} not found in cache")
                
        return feature_activations
    
    def compute_japanese_feature_reward(self, feature_activations, attention_mask):
        """
        Compute reward based on Japanese feature activations.
        
        Args:
            feature_activations: Feature activations from SAE [batch_size, seq_len, n_features]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Reward tensor [batch_size]
        """
        batch_size, seq_len, _ = feature_activations.shape
        
        # Extract only Japanese features
        ja_activations = torch.zeros(
            (batch_size, seq_len, len(self.ja_feature_indices)),
            device=self.device
        )
        
        for i, idx in enumerate(self.ja_feature_indices):
            if idx < feature_activations.shape[2]:
                ja_activations[:, :, i] = feature_activations[:, :, idx]
            else:
                logger.warning(f"Feature index {idx} out of bounds (max={feature_activations.shape[2]-1})")
        
        # Apply attention mask to only consider non-padding tokens
        masked_activations = ja_activations * attention_mask.unsqueeze(-1)
        
        # Calculate mean activation per token, then sum across sequence
        # This rewards both the activation strength and the number of tokens with Japanese features
        token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        mean_activations = masked_activations.sum(dim=1) / token_counts
        
        # Mean across features (single reward per sequence)
        sequence_rewards = mean_activations.mean(dim=1)
        
        return sequence_rewards
    
    def get_value(self, hidden_states, attention_mask):
        """
        Get value estimate from hidden states.
        
        Args:
            hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Value tensor [batch_size]
        """
        # Apply attention mask to get mean of only valid tokens
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
        token_counts = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        mean_hidden = masked_hidden.sum(dim=1) / token_counts
        
        # Get value estimate
        value = self.value_head(mean_hidden)
        return value.squeeze(-1)
    
    def train_step(self, input_ids, attention_mask):
        """
        Perform one PPO training step
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Dictionary of training statistics
        """
        self.model.train()
        self.value_head.train()
        
        # STAGE 1: Reference model forward pass (to get old probs and compute rewards)
        with torch.no_grad():
            # Get SAE feature activations for reward calculation
            feature_activations = self.get_sae_feature_activations(input_ids, attention_mask)
            
            # Compute rewards from Japanese features
            rewards = self.compute_japanese_feature_reward(feature_activations, attention_mask)
            
            # Run reference model forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            old_logits = outputs.logits
            old_hidden = outputs.hidden_states[-1]
            
            # Calculate old action log probs
            old_log_probs = torch.log_softmax(old_logits[:, :-1], dim=-1)
            old_action_log_probs = torch.gather(
                old_log_probs, -1, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            
            # Calculate mask for padding tokens
            valid_tokens_mask = attention_mask[:, 1:] > 0
            old_action_log_probs = old_action_log_probs * valid_tokens_mask
            
            # Get old values
            old_values = self.get_value(old_hidden, attention_mask)
        
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
        values = self.get_value(hidden_states, attention_mask)
        
        # Compute advantages
        advantages = rewards - old_values
        
        # Normalize advantages for stable training
        if advantages.shape[0] > 1:  # Only normalize if batch size > 1
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
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
        token_counts = valid_tokens_mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
        entropy = masked_entropy.sum(dim=1) / token_counts
        entropy_loss = -self.c2 * entropy.mean()
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
        self.optimizer.step()
        
        # Calculate statistics for logging
        stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "mean_value": values.mean().item(),
            "mean_ratio": ratio.mean().item(),
            "clip_fraction": (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item(),
            "approx_kl": (old_sequence_log_probs - sequence_log_probs).mean().item(),
            "ja_feature_activation": rewards.mean().item(),  # Log the Japanese feature activation as well
        }
        
        return stats
    
    def evaluate(self, dataloader):
        """
        Evaluate the model on a dataset
        
        Args:
            dataloader: DataLoader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        self.value_head.eval()
        
        total_loss = 0
        total_ja_activation = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Get model outputs
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["input_ids"],
                )
                
                # Get loss
                loss = outputs.loss
                
                # Get SAE feature activations
                feature_activations = self.get_sae_feature_activations(
                    batch["input_ids"], 
                    batch["attention_mask"]
                )
                
                # Calculate Japanese feature activation
                ja_activation = self.compute_japanese_feature_reward(
                    feature_activations, 
                    batch["attention_mask"]
                ).mean().item()
                
                # Update statistics
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_ja_activation += ja_activation * batch_size
                total_samples += batch_size
        
        # Calculate metrics
        avg_loss = total_loss / total_samples
        avg_ja_activation = total_ja_activation / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_ja_activation": avg_ja_activation,
        }
    
    def generate_text(self, prompt, max_length=50, temperature=0.7):
        """
        Generate text from a prompt and analyze Japanese feature activation
        
        Args:
            prompt: Text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with generated text and analysis
        """
        self.model.eval()
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode text
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Analyze Japanese feature activation
        with torch.no_grad():
            feature_activations = self.get_sae_feature_activations(
                output_ids, 
                torch.ones_like(output_ids, device=self.device)
            )
            
            ja_activations = torch.zeros(
                (feature_activations.shape[0], feature_activations.shape[1], len(self.ja_feature_indices)),
                device=self.device
            )
            
            for i, idx in enumerate(self.ja_feature_indices):
                ja_activations[:, :, i] = feature_activations[:, :, idx]
                
            # Calculate token-wise activation
            token_activations = ja_activations.mean(dim=2).squeeze().cpu().numpy()
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(output_ids[0])
            
            # Create activation analysis
            activation_analysis = [
                {"token": tokens[i], "ja_activation": float(token_activations[i])}
                for i in range(len(tokens))
            ]
        
        return {
            "prompt": prompt,
            "generated_text": generated_text,
            "overall_ja_activation": float(token_activations.mean()),
            "token_activations": activation_analysis,
        }
    
    def train(
        self,
        dataset_path: str = "assets/dataset/flores200",
        num_epochs: int = 3,
        eval_steps: int = 500,
        save_steps: int = 500,
        log_steps: int = 100,
        generation_prompts: List[str] = None,
    ):
        """
        Train the model using PPO
        
        Args:
            dataset_path: Path to dataset
            num_epochs: Number of training epochs
            eval_steps: Number of steps between evaluations
            save_steps: Number of steps between model saves
            log_steps: Number of steps between generation logging
            generation_prompts: List of prompts for text generation
        """
        # Create datasets
        logger.info(f"Creating datasets from {dataset_path}...")
        train_dataset = JapaneseTextDataset(
            dataset_path=dataset_path,
            tokenizer=self.tokenizer,
            split="train",
            max_length=self.max_length,
            text_column="ja",
        )
        
        eval_dataset = JapaneseTextDataset(
            dataset_path=dataset_path,
            tokenizer=self.tokenizer,
            split="dev",
            max_length=self.max_length,
            text_column="ja",
            sample_size=50,  # Use smaller eval dataset for speed
        )
        
        # Create dataloaders
        train_dataloader = get_dataloader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        eval_dataloader = get_dataloader(
            dataset=eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )
        
        # Set up generation prompts if not provided
        if generation_prompts is None:
            generation_prompts = [
                "日本語でこの文章を続けてください。",  # Please continue this sentence in Japanese
                "今日の天気は",  # Today's weather is
                "日本の文化について",  # About Japanese culture
                "東京は",  # Tokyo is
            ]
        
        # Calculate total steps
        total_steps = len(train_dataloader) * num_epochs
        logger.info(f"Starting training for {num_epochs} epochs ({total_steps} total steps)")
        
        # Training loop
        global_step = 0
        
        for epoch in range(num_epochs):
            epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(epoch_iterator):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Perform training step
                train_stats = self.train_step(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                
                # Update step
                global_step += 1
                self.tracker.update_step(global_step)
                
                # Log training stats
                self.tracker.log_metrics(train_stats)
                
                # Log to W&B
                if self.use_wandb:
                    wandb.log(train_stats, step=global_step)
                
                # Update progress bar
                desc = f"Epoch {epoch+1}/{num_epochs} [Loss: {train_stats['total_loss']:.4f}] [JA: {train_stats['ja_feature_activation']:.4f}]"
                epoch_iterator.set_description(desc)
                
                # Log generations
                if global_step % log_steps == 0:
                    self._log_generations(generation_prompts, global_step)
                
                # Evaluate
                if global_step % eval_steps == 0:
                    logger.info(f"Evaluating at step {global_step}...")
                    eval_metrics = self.evaluate(eval_dataloader)
                    
                    # Log evaluation metrics
                    self.tracker.log_metrics(eval_metrics, split="eval")
                    
                    if self.use_wandb:
                        wandb.log(eval_metrics, step=global_step)
                
                # Save model
                if global_step % save_steps == 0:
                    logger.info(f"Saving model at step {global_step}...")
                    
                    # Check if this is the best model
                    if hasattr(self.tracker, "best_ja_activation"):
                        current_ja = train_stats["ja_feature_activation"]
                        if current_ja > self.tracker.best_ja_activation:
                            self.tracker.best_ja_activation = current_ja
                            self._save_checkpoint(is_best=True)
                        else:
                            self._save_checkpoint(is_best=False)
                    else:
                        self.tracker.best_ja_activation = train_stats["ja_feature_activation"]
                        self._save_checkpoint(is_best=True)
            
            # End of epoch evaluation
            logger.info(f"End of epoch {epoch+1} evaluation...")
            eval_metrics = self.evaluate(eval_dataloader)
            self.tracker.log_metrics(eval_metrics, split="eval")
            
            if self.use_wandb:
                wandb.log(eval_metrics, step=global_step)
            
            # Save end of epoch model
            logger.info(f"Saving model at end of epoch {epoch+1}...")
            self._save_checkpoint(is_best=False, suffix=f"epoch_{epoch+1}")
        
        # Final save
        logger.info("Training complete. Saving final model...")
        self._save_checkpoint(is_best=False, suffix="final")
        
        # Plot training metrics
        metrics_plot_path = os.path.join(self.output_dir, "training_metrics.png")
        self.tracker.plot_metrics(output_file=metrics_plot_path)
        
        if self.use_wandb:
            wandb.log({"training_metrics": wandb.Image(metrics_plot_path)})
            wandb.finish()
    
    def _log_generations(self, prompts, step):
        """
        Generate and log text from prompts
        
        Args:
            prompts: List of prompts
            step: Current training step
        """
        logger.info(f"Generating text samples at step {step}...")
        generations = []
        
        for prompt in prompts:
            generation = self.generate_text(prompt)
            generations.append(generation)
            
            # Log to console
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generation['generated_text']}")
            logger.info(f"Japanese Activation: {generation['overall_ja_activation']:.4f}\n")
        
        # Log to W&B
        if self.use_wandb:
            for i, gen in enumerate(generations):
                wandb.log({
                    f"generation_{i}": {
                        "prompt": gen["prompt"],
                        "text": gen["generated_text"],
                        "ja_activation": gen["overall_ja_activation"],
                    }
                }, step=step)
                
                # Also log token activation as a table
                if len(gen["token_activations"]) > 0:
                    activation_table = wandb.Table(
                        columns=["token", "ja_activation"],
                        data=[[item["token"], item["ja_activation"]] for item in gen["token_activations"]]
                    )
                    wandb.log({f"token_activations_{i}": activation_table}, step=step)
    
    def _save_checkpoint(self, is_best=False, suffix=None):
        """
        Save model checkpoint
        
        Args:
            is_best: Whether this is the best model so far
            suffix: Optional suffix for the checkpoint name
        """
        # Determine checkpoint name
        if is_best:
            checkpoint_name = "best_model"
        elif suffix:
            checkpoint_name = f"checkpoint_{suffix}"
        else:
            checkpoint_name = f"checkpoint_{self.tracker.step}"
        
        # Save checkpoint
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save value head and other data
        extra_state = {
            "value_head": self.value_head.state_dict(),
            "step": self.tracker.step,
            "ja_feature_indices": self.ja_feature_indices.cpu(),
        }
        
        torch.save(extra_state, checkpoint_dir / "extra_state.pt")
        logger.info(f"Model saved to {checkpoint_dir}")


def main():
    """Main function to run the training"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a model to optimize for Japanese language features")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, default="assets/gpt2-small",
                        help="Path to pretrained model")
    parser.add_argument("--features_path", type=str, default="assets/features/lang_features.json",
                        help="Path to language features file")
    
    # Training arguments
    parser.add_argument("--dataset_path", type=str, default="assets/dataset/flores200",
                        help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="assets/ppo_model",
                        help="Output directory for model and logs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="PPO clipping parameter")
    
    # Evaluation arguments
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save model every N steps")
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log generations every N steps")
    
    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="language-feature-rl",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable W&B logging")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (None for auto)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Initialize trainer
    trainer = PPOTrainer(
        model_path=args.model_path,
        features_path=args.features_path,
        device=device,
        learning_rate=args.learning_rate,
        clip_epsilon=args.clip_epsilon,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )
    
    # Train the model
    trainer.train(
        dataset_path=args.dataset_path,
        num_epochs=args.num_epochs,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )


if __name__ == "__main__":
    main()