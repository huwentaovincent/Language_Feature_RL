import os
import torch
import json
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from sae_lens import SAE, HookedSAETransformer

# Set device
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

def get_language_features():
    # Create assets directory if it doesn't exist
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Load FLORES200 dataset
    print("Loading FLORES200 dataset...")
    dataset = load_dataset("Muennighoff/flores200", trust_remote_code=True)
    
    # Initialize model and SAE
    print("Initializing model and SAE...")
    model = HookedSAETransformer.from_pretrained("gpt2-small", device=device)
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.7.hook_resid_pre",
        device=device,
    )
    
    # Languages to process
    languages = ["zh", "ja", "es", "en", "de", "fr"]
    features_per_lang = 100
    
    # Dictionary to store features for each language
    lang_features = {}
    
    # Process each language
    for lang in languages:
        print(f"\nProcessing {lang}...")
        
        # Get text samples for the language
        lang_samples = dataset["dev"][lang][:100]  # Use 100 samples for each language
        
        # Store activations for all samples
        all_activations = []
        
        # Process each sample
        for text in tqdm(lang_samples, desc=f"Processing {lang} samples"):
            # Run model with SAE
            _, cache = model.run_with_cache_with_saes(text, saes=[sae])
            
            # Get feature activations
            feature_acts = cache["blocks.7.hook_resid_pre.hook_sae_acts_post"]
            
            # Average across sequence length and batch
            avg_acts = feature_acts.mean(dim=[0, 1]).cpu()
            all_activations.append(avg_acts)
        
        # Average across all samples
        mean_activations = torch.stack(all_activations).mean(dim=0)
        
        # Get top features
        _, top_indices = torch.topk(mean_activations, features_per_lang)
        
        # Store features for this language
        lang_features[lang] = {
            "feature_indices": top_indices.tolist(),
            "mean_activations": mean_activations[top_indices].tolist()
        }
    
    # Save results to JSON file
    output_file = assets_dir / "lang_features.json"
    with open(output_file, "w") as f:
        json.dump(lang_features, f, indent=2)
    
    print(f"\nFeatures saved to {output_file}")

if __name__ == "__main__":
    get_language_features()

