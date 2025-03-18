import os
import torch
import json
import argparse
import logging
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("get_lang_features")

class LanguageFeatureExtractor:
    def __init__(
        self,
        sae_release="gpt2-small-res-jb",
        sae_id="blocks.7.hook_resid_pre",
        model_name="gpt2-small",
        dataset_path="assets/dataset/flores200",
        output_dir="assets/features",
        device=None,
        features_per_lang=100,
        sample_size=100,
        batch_size=4,
        max_length=128,
    ):
        """
        Initialize the language feature extractor.
        
        Args:
            sae_release: SAE release name
            sae_id: SAE ID for specific hook point
            model_name: Model to use
            dataset_path: Path to FLORES200 dataset
            output_dir: Directory to save features
            device: Device to use (None for auto-detect)
            features_per_lang: Number of top features to extract per language
            sample_size: Number of samples to use per language
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        self.sae_release = sae_release
        self.sae_id = sae_id
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.features_per_lang = features_per_lang
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model and SAE
        self._init_model_and_sae()
        
        # Cache to store results
        self.cache = {}
    
    def _init_model_and_sae(self):
        """Initialize the model and SAE"""
        try:
            logger.info(f"Initializing model {self.model_name}...")
            self.model = HookedSAETransformer.from_pretrained(self.model_name, device=self.device)
            
            logger.info(f"Loading SAE {self.sae_release}/{self.sae_id}...")
            self.sae, self.cfg_dict, self.sparsity = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=self.sae_id,
                device=self.device,
            )
            
            # Log some info about the SAE
            logger.info(f"SAE input dimension: {self.sae.cfg.d_in}")
            logger.info(f"SAE feature dimension: {self.sae.cfg.d_sae}")
            logger.info(f"SAE hook point: {self.sae.cfg.hook_name}")
            
        except Exception as e:
            logger.error(f"Error initializing model and SAE: {str(e)}")
            raise
    
    def _load_dataset(self, languages, split="dev"):
        """Load dataset for specified languages"""
        try:
            logger.info(f"Loading dataset from {self.dataset_path}...")
            
            # Try to load from disk first
            if os.path.exists(self.dataset_path):
                try:
                    dataset = load_from_disk(os.path.join(self.dataset_path, split))
                    logger.info(f"Loaded {split} split from disk")
                except Exception as disk_error:
                    logger.warning(f"Failed to load from disk: {str(disk_error)}")
                    logger.info("Trying to load from HuggingFace...")
                    dataset = load_dataset("Muennighoff/flores200", split=split, trust_remote_code=True)
            else:
                # Load from HuggingFace
                dataset = load_dataset("Muennighoff/flores200", split=split, trust_remote_code=True)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def _process_language_batch(self, texts, language):
        """Process a batch of texts for a single language"""
        # Tokenize texts
        all_activations = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Run model with SAE
            _, cache = self.model.run_with_cache_with_saes(batch_texts, saes=[self.sae])
            
            # Get feature activations
            feature_acts = cache[f"{self.sae.cfg.hook_name}.hook_sae_acts_post"]
            
            # Average across sequence length and batch 
            # (shape: [batch_size, seq_len, features] -> [features])
            for j in range(feature_acts.shape[0]):
                avg_acts = feature_acts[j].mean(dim=0).cpu()
                all_activations.append(avg_acts)
        
        # Stack all activations
        stacked_activations = torch.stack(all_activations)
        
        # Average across all samples
        mean_activations = stacked_activations.mean(dim=0)
        
        # Also compute standard deviation to measure reliability
        std_activations = stacked_activations.std(dim=0)
        
        # Get top features
        values, top_indices = torch.topk(mean_activations, self.features_per_lang)
        
        # Get their standard deviations
        top_stds = std_activations[top_indices]
        
        # Store in cache
        self.cache[language] = {
            "feature_indices": top_indices.tolist(),
            "mean_activations": values.tolist(),
            "std_activations": top_stds.tolist(),
            "activation_stats": {
                "global_mean": mean_activations.mean().item(),
                "global_std": mean_activations.std().item(),
                "sparsity": (mean_activations > 0).float().mean().item(),
            }
        }
        
        return self.cache[language]
    
    def extract_language_features(self, languages=None):
        """
        Extract language-specific features for specified languages.
        
        Args:
            languages: List of language codes (None for default set)
            
        Returns:
            Dictionary mapping languages to feature information
        """
        # Default languages
        if languages is None:
            languages = ["zh", "ja", "es", "en", "de", "fr"]
        
        logger.info(f"Extracting features for languages: {', '.join(languages)}")
        
        # Load dataset
        dataset = self._load_dataset(languages)
        
        # Process each language
        for lang in languages:
            logger.info(f"\nProcessing {lang}...")
            
            try:
                # Get text samples for the language
                if lang in dataset:
                    lang_samples = [item for item in dataset if lang in item][:self.sample_size]
                    texts = [item[lang] for item in lang_samples]
                    
                    if not texts:
                        logger.warning(f"No texts found for language {lang}, skipping.")
                        continue
                    
                    logger.info(f"Processing {len(texts)} samples for {lang}...")
                    
                    # Process the language
                    features = self._process_language_batch(texts, lang)
                    
                    logger.info(f"Extracted {len(features['feature_indices'])} features for {lang}")
                    logger.info(f"Top feature activations: {features['mean_activations'][:5]}")
                    logger.info(f"Activation sparsity: {features['activation_stats']['sparsity']:.2%}")
                else:
                    logger.warning(f"Language {lang} not found in dataset, skipping.")
            except Exception as e:
                logger.error(f"Error processing language {lang}: {str(e)}")
                # Continue with other languages
                continue
        
        # Return all extracted features
        return self.cache
    
    def analyze_feature_overlap(self):
        """Analyze overlap between language features"""
        if not self.cache:
            logger.warning("No features extracted yet. Call extract_language_features() first.")
            return {}
        
        # Count feature occurrences across languages
        feature_counts = defaultdict(int)
        feature_langs = defaultdict(list)
        
        for lang, data in self.cache.items():
            for feature in data["feature_indices"]:
                feature_counts[feature] += 1
                feature_langs[feature].append(lang)
        
        # Compute overlap statistics
        total_unique_features = len(feature_counts)
        features_by_count = defaultdict(list)
        
        for feature, count in feature_counts.items():
            features_by_count[count].append(feature)
        
        # Compute overlap matrix
        languages = list(self.cache.keys())
        overlap_matrix = np.zeros((len(languages), len(languages)))
        
        for i, lang1 in enumerate(languages):
            features1 = set(self.cache[lang1]["feature_indices"])
            for j, lang2 in enumerate(languages):
                features2 = set(self.cache[lang2]["feature_indices"])
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    overlap = len(features1.intersection(features2))
                    union = len(features1.union(features2))
                    overlap_matrix[i, j] = overlap / union
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot overlap matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(overlap_matrix, annot=True, fmt=".2f", xticklabels=languages, yticklabels=languages)
        plt.title("Feature Overlap Between Languages (Jaccard Index)")
        plt.tight_layout()
        plt.savefig(plots_dir / "language_feature_overlap.png")
        
        # Return overlap analysis
        return {
            "total_unique_features": total_unique_features,
            "features_by_count": {k: len(v) for k, v in features_by_count.items()},
            "features_in_common": {
                f"in_{k}_languages": len(v) for k, v in features_by_count.items() if k > 1
            },
            "overlap_matrix": overlap_matrix.tolist(),
            "languages": languages,
        }
    
    def save_features(self, filename="lang_features.json"):
        """Save extracted features to JSON file"""
        if not self.cache:
            logger.warning("No features extracted yet. Call extract_language_features() first.")
            return
        
        output_file = self.output_dir / filename
        
        # Add metadata
        output_data = {
            "metadata": {
                "sae_release": self.sae_release,
                "sae_id": self.sae_id,
                "model_name": self.model_name,
                "features_per_language": self.features_per_lang,
                "sample_size": self.sample_size,
                "extraction_date": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
            },
            "languages": self.cache,
        }
        
        # Save to file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Features saved to {output_file}")
        
        # Also save analysis
        analysis = self.analyze_feature_overlap()
        
        analysis_file = self.output_dir / "feature_analysis.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Feature analysis saved to {analysis_file}")


def main():
    """Main function to extract language features"""
    parser = argparse.ArgumentParser(description="Extract language-specific features using SAE")
    
    # Model configuration
    parser.add_argument("--sae_release", type=str, default="gpt2-small-res-jb",
                        help="SAE release name")
    parser.add_argument("--sae_id", type=str, default="blocks.7.hook_resid_pre",
                        help="SAE ID for specific hook point")
    parser.add_argument("--model_name", type=str, default="gpt2-small",
                        help="Model to use")
    
    # Dataset configuration
    parser.add_argument("--dataset_path", type=str, default="assets/dataset/flores200",
                        help="Path to FLORES200 dataset")
    parser.add_argument("--languages", type=str, nargs="+", 
                        default=["zh", "ja", "es", "en", "de", "fr"],
                        help="Languages to process")
    parser.add_argument("--split", type=str, default="dev",
                        help="Dataset split to use")
    
    # Feature extraction parameters
    parser.add_argument("--features_per_lang", type=int, default=100,
                        help="Number of top features to extract per language")
    parser.add_argument("--sample_size", type=int, default=100,
                        help="Number of samples to use per language")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="assets/features",
                        help="Directory to save features")
    parser.add_argument("--output_file", type=str, default="lang_features.json",
                        help="Output filename")
    
    # Device configuration
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (None for auto-detect)")
    
    args = parser.parse_args()
    
    # Initialize feature extractor
    extractor = LanguageFeatureExtractor(
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        device=args.device,
        features_per_lang=args.features_per_lang,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    
    # Extract features
    extractor.extract_language_features(args.languages)
    
    # Save features
    extractor.save_features(args.output_file)


if __name__ == "__main__":
    main()