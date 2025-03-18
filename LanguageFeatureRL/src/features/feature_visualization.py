# src/features/feature_visualization.py
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualize_language_features(features_path, output_dir):
    """Create visualizations of language-specific features"""
    with open(features_path, 'r') as f:
        lang_features = json.load(f)
    
    # Create feature count visualization
    plt.figure(figsize=(10, 6))
    counts = {lang: len(data["feature_indices"]) for lang, data in lang_features.items()}
    sns.barplot(x=list(counts.keys()), y=list(counts.values()))
    plt.title("Feature Count by Language")
    plt.savefig(Path(output_dir) / "feature_counts.png")
    
    # Create feature overlap heatmap
    # Create activation strength comparison
    # etc.

def main():
    visualize_language_features(
        "assets/features/filtered_lang_features.json",
        "assets/results/visualizations"
    )

if __name__ == "__main__":
    main()