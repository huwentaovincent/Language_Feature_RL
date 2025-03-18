import json
from collections import Counter
from pathlib import Path

def filter_shared_features():
    # Load the original features
    input_file = Path("assets/lang_features.json")
    with open(input_file, "r") as f:
        lang_features = json.load(f)
    
    # Count feature occurrences across languages
    feature_counts = Counter()
    for lang_data in lang_features.values():
        feature_counts.update(lang_data["feature_indices"])
    
    # Find features that appear in more than 3 languages
    shared_features = {feature for feature, count in feature_counts.items() if count > 3}
    
    print(f"Found {len(shared_features)} features shared across more than 3 languages")
    
    # Filter out shared features for each language
    filtered_features = {}
    for lang, data in lang_features.items():
        # Get indices of features that are not shared
        original_indices = data["feature_indices"]
        original_activations = data["mean_activations"]
        
        filtered_indices = []
        filtered_activations = []
        
        for idx, act in zip(original_indices, original_activations):
            if idx not in shared_features:
                filtered_indices.append(idx)
                filtered_activations.append(act)
        
        filtered_features[lang] = {
            "feature_indices": filtered_indices,
            "mean_activations": filtered_activations
        }
        
        print(f"{lang}: Removed {len(original_indices) - len(filtered_indices)} shared features")
    
    # Save filtered features
    output_file = Path("assets/filtered_lang_features.json")
    with open(output_file, "w") as f:
        json.dump(filtered_features, f, indent=2)
    
    print(f"\nFiltered features saved to {output_file}")
    
    # Print some statistics
    print("\nFeature counts per language:")
    for lang, data in filtered_features.items():
        print(f"{lang}: {len(data['feature_indices'])} features")

if __name__ == "__main__":
    filter_shared_features() 