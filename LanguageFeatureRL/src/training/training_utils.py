import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader
import random
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("training_utils")


class TrainingTracker:
    """Tracks training metrics and handles model checkpointing."""
    
    def __init__(
        self,
        output_dir: str,
        model_name: str = "model",
        save_every: int = 1000,
        log_every: int = 10,
    ):
        """
        Initialize the training tracker.
        
        Args:
            output_dir: Directory to save checkpoints and logs
            model_name: Name to use for saved model checkpoints
            save_every: Save model every N steps
            log_every: Log training metrics every N steps
        """
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.save_every = save_every
        self.log_every = log_every
        
        # Make directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics = {
            "train": {},
            "eval": {},
        }
        self.step = 0
        self.best_metric = float('inf')
        self.best_step = 0
        
        # Initialize log file
        self.log_file = self.log_dir / "training_log.jsonl"
        
        # Initialize training start time
        self.start_time = datetime.now()
        logger.info(f"Training tracker initialized. Output directory: {self.output_dir}")
    
    def update_step(self, step: int) -> None:
        """Update the current training step."""
        self.step = step
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, split: str = "train") -> None:
        """
        Log metrics for the current step.
        
        Args:
            metrics: Dictionary of metric name to value
            step: Current step (if None, use self.step)
            split: Data split, either "train" or "eval"
        """
        current_step = step if step is not None else self.step
        
        # Add metrics to tracking
        if split not in self.metrics:
            self.metrics[split] = {}
            
        for k, v in metrics.items():
            if k not in self.metrics[split]:
                self.metrics[split][k] = []
            self.metrics[split][k].append((current_step, v))
        
        # Log to file if needed
        if current_step % self.log_every == 0 or split == "eval":
            log_dict = {
                "step": current_step,
                "split": split,
                "metrics": metrics,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": (datetime.now() - self.start_time).total_seconds(),
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_dict) + "\n")
                
            # Print metrics
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Step {current_step} ({split}): {metrics_str}")
    
    def should_save_checkpoint(self, step: Optional[int] = None) -> bool:
        """Check if we should save a checkpoint at the current step."""
        current_step = step if step is not None else self.step
        return current_step > 0 and current_step % self.save_every == 0
    
    def is_best_model(self, metric_value: float, metric_name: str = "loss", lower_is_better: bool = True) -> bool:
        """Check if the current model is the best seen so far based on a metric."""
        if lower_is_better:
            is_best = metric_value < self.best_metric
        else:
            is_best = metric_value > self.best_metric
        
        if is_best:
            self.best_metric = metric_value
            self.best_step = self.step
            logger.info(f"New best model with {metric_name}: {metric_value:.4f} at step {self.step}")
        
        return is_best
    
    def save_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer = None,
        scheduler: Any = None,
        is_best: bool = False,
        extra_state: Dict[str, Any] = None,
    ) -> None:
        """
        Save a model checkpoint.
        
        Args:
            model: The model to save
            tokenizer: The tokenizer to save
            optimizer: The optimizer to save (optional)
            scheduler: The scheduler to save (optional)
            is_best: Whether this is the best model so far
            extra_state: Additional state to save
        """
        # Checkpoint filename
        if is_best:
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best"
        else:
            checkpoint_path = self.checkpoint_dir / f"{self.model_name}_step_{self.step}"
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save optimizer, scheduler, and extra state
        state_dict = {
            "step": self.step,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
        }
        
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        
        if scheduler is not None:
            state_dict["scheduler"] = scheduler.state_dict()
        
        if extra_state is not None:
            state_dict.update(extra_state)
        
        torch.save(state_dict, checkpoint_path / "training_state.pt")
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a model checkpoint.
        
        Args:
            model: The model to load into
            tokenizer: The tokenizer to load into
            optimizer: The optimizer to load into (optional)
            scheduler: The scheduler to load into (optional)
            checkpoint_path: Path to checkpoint (if None, use latest or best)
            load_best: Whether to load the best model instead of latest
            
        Returns:
            Dictionary of extra state
        """
        # Find checkpoint path if not provided
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best"
            else:
                # Find highest step checkpoint
                checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_step_*"))
                if not checkpoints:
                    logger.warning("No checkpoints found to load.")
                    return {}
                
                # Parse step numbers and find highest
                step_nums = [int(cp.name.split("_")[-1]) for cp in checkpoints]
                latest_idx = np.argmax(step_nums)
                checkpoint_path = checkpoints[latest_idx]
        else:
            checkpoint_path = Path(checkpoint_path)
        
        # Check if checkpoint exists
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint {checkpoint_path} does not exist.")
            return {}
        
        # Load model and tokenizer
        model.from_pretrained(checkpoint_path)
        tokenizer.from_pretrained(checkpoint_path)
        
        # Load optimizer, scheduler, and extra state
        state_dict_path = checkpoint_path / "training_state.pt"
        if state_dict_path.exists():
            state_dict = torch.load(state_dict_path, map_location="cpu")
            
            # Update step and best metrics
            self.step = state_dict.get("step", 0)
            self.best_metric = state_dict.get("best_metric", float('inf'))
            self.best_step = state_dict.get("best_step", 0)
            
            # Load optimizer state
            if optimizer is not None and "optimizer" in state_dict:
                optimizer.load_state_dict(state_dict["optimizer"])
            
            # Load scheduler state
            if scheduler is not None and "scheduler" in state_dict:
                scheduler.load_state_dict(state_dict["scheduler"])
            
            # Remove known keys to get extra state
            for key in ["step", "best_metric", "best_step", "optimizer", "scheduler"]:
                if key in state_dict:
                    del state_dict[key]
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return state_dict
        else:
            logger.warning(f"No training state found at {state_dict_path}")
            return {}
    
    def plot_metrics(self, output_file: Optional[str] = None) -> None:
        """
        Plot training metrics.
        
        Args:
            output_file: Path to save the plot (if None, display only)
        """
        plt.figure(figsize=(12, 8))
        
        # Get all unique metrics
        all_metrics = set()
        for split_metrics in self.metrics.values():
            all_metrics.update(split_metrics.keys())
        
        # Create subplots
        n_plots = len(all_metrics)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        for i, metric_name in enumerate(sorted(all_metrics)):
            plt.subplot(n_rows, n_cols, i + 1)
            
            for split, split_metrics in self.metrics.items():
                if metric_name in split_metrics:
                    steps, values = zip(*split_metrics[metric_name])
                    plt.plot(steps, values, label=f"{split}")
            
            plt.title(metric_name)
            plt.xlabel("Step")
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file)
            logger.info(f"Saved metrics plot to {output_file}")
        else:
            plt.show()


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    logger.info(f"Random seed set to {seed}")


class JapaneseTextDataset(Dataset):
    """Dataset for Japanese text from FLORES200."""
    
    def __init__(
        self, 
        dataset_path: str,
        tokenizer: PreTrainedTokenizer,
        split: str = "train",
        max_length: int = 128,
        text_column: str = "ja",
        seed: int = 42,
        sample_size: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset_path: Path to FLORES200 dataset
            tokenizer: Tokenizer to use
            split: Dataset split to use
            max_length: Maximum sequence length
            text_column: Column name containing Japanese text
            seed: Random seed for sampling
            sample_size: Number of samples to use (if None, use all)
        """
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # Load dataset
        from datasets import load_from_disk, load_dataset
        
        try:
            if os.path.exists(dataset_path):
                self.dataset = load_from_disk(dataset_path)
            else:
                self.dataset = load_dataset(dataset_path, trust_remote_code=True)
            
            # Get specified split
            if split in self.dataset:
                self.data = self.dataset[split]
            else:
                logger.warning(f"Split {split} not found. Using 'train' instead.")
                self.data = self.dataset["train"]
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
        
        # Filter to get only entries with Japanese text
        filtered_data = []
        for item in self.data:
            if self.text_column in item and item[self.text_column]:
                filtered_data.append({"text": item[self.text_column]})
        
        # Sample if requested
        if sample_size and sample_size < len(filtered_data):
            random.seed(seed)
            filtered_data = random.sample(filtered_data, sample_size)
        
        self.examples = filtered_data
        logger.info(f"Loaded {len(self.examples)} examples from {dataset_path}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]["text"]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        # Remove batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        return item


def pad_sequence(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of examples to the same length.
    
    Args:
        batch: List of dictionaries with tensors
        
    Returns:
        Dictionary with padded tensors
    """
    # Get keys from first example
    keys = batch[0].keys()
    
    # Get max length for each key
    max_lengths = {}
    for key in keys:
        if batch[0][key].dim() >= 1:
            max_lengths[key] = max(x[key].size(0) for x in batch)
    
    # Pad each example
    padded_batch = {key: [] for key in keys}
    
    for example in batch:
        for key in keys:
            tensor = example[key]
            
            if key in max_lengths:
                # Pad sequence
                pad_length = max_lengths[key] - tensor.size(0)
                if pad_length > 0:
                    if tensor.dim() == 1:
                        tensor = torch.cat([tensor, torch.zeros(pad_length, dtype=tensor.dtype)])
                    else:
                        tensor = torch.cat([tensor, torch.zeros(pad_length, *tensor.size()[1:], dtype=tensor.dtype)])
            
            padded_batch[key].append(tensor)
    
    # Stack tensors
    for key in keys:
        padded_batch[key] = torch.stack(padded_batch[key])
    
    return padded_batch


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a dataloader for a dataset.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for dataloader
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_sequence,
    )


def compute_perplexity(
    model: PreTrainedModel,
    dataloader: DataLoader,
    device: str = "cuda"
) -> float:
    """
    Compute the perplexity of a model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with examples
        device: Device to use
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Create labels (shifted input_ids)
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100  # Ignore last token since we don't have its next token
            
            # Where attention mask is 0, set label to -100 to ignore
            labels = labels * attention_mask - 100 * (1 - attention_mask)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            total_loss += loss.item() * torch.sum(attention_mask).item()
            total_tokens += torch.sum(attention_mask).item()
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def compute_feature_activation_score(
    model: Union[PreTrainedModel, torch.nn.Module],
    dataloader: DataLoader,
    feature_indices: List[int],
    device: str = "cuda",
    sae_model_getter: Optional[callable] = None
) -> Dict[str, float]:
    """
    Compute how strongly specific features activate on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with examples
        feature_indices: List of feature indices to track
        device: Device to use
        sae_model_getter: Function to get SAE model from regular model
        
    Returns:
        Dictionary with activation statistics
    """
    if sae_model_getter is None:
        # Assume model has a sae_model attribute
        sae_model = getattr(model, "sae_model", model)
    else:
        sae_model = sae_model_getter(model)
    
    # Ensure sae_model has the necessary methods for feature extraction
    assert hasattr(sae_model, "run_with_cache_with_saes"), "Model doesn't support SAE feature extraction"
    
    # Get the SAE
    sae = getattr(sae_model, "sae", None)
    if sae is None:
        logger.error("No SAE found on model")
        return {}
    
    feature_indices = torch.tensor(feature_indices, device=device)
    all_activations = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Run model with SAE
            _, cache = sae_model.run_with_cache_with_saes(
                input_ids=input_ids,
                attention_mask=attention_mask,
                saes=[sae]
            )
            
            # Get feature activations (assuming a standard hook point name)
            feature_acts = cache.get("blocks.7.hook_resid_pre.hook_sae_acts_post", None)
            if feature_acts is None:
                logger.error("Feature activations not found in cache")
                return {}
            
            # Extract activations for targeted features
            batch_activations = torch.zeros(
                (feature_acts.shape[0], feature_acts.shape[1], len(feature_indices)),
                device=device
            )
            
            for i, idx in enumerate(feature_indices):
                batch_activations[:, :, i] = feature_acts[:, :, idx]
            
            # Average across sequence and features
            masked_activations = batch_activations * attention_mask.unsqueeze(-1)
            seq_lengths = attention_mask.sum(dim=1, keepdim=True)
            avg_activations = (masked_activations.sum(dim=1) / seq_lengths).mean(dim=1)
            
            all_activations.append(avg_activations)
    
    if not all_activations:
        return {}
    
    # Combine results
    all_activations = torch.cat(all_activations)
    
    return {
        "mean_activation": all_activations.mean().item(),
        "median_activation": all_activations.median().item(),
        "max_activation": all_activations.max().item(),
        "min_activation": all_activations.min().item(),
        "std_activation": all_activations.std().item(),
        "non_zero_percentage": (all_activations > 0).float().mean().item() * 100,
    }