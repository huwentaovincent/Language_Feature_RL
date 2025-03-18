import os
from datasets import load_dataset
from pathlib import Path

def prepare_flores200():
    # Create dataset directory if it doesn't exist
    dataset_dir = Path("assets/dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading FLORES200 dataset...")
    # Load the dataset with trust_remote_code=True
    dataset = load_dataset("Muennighoff/flores200", trust_remote_code=True)
    
    # Save the dataset locally
    print("Saving dataset to assets/dataset...")
    dataset.save_to_disk(str(dataset_dir / "flores200"))
    
    print("Dataset preparation completed!")
    print(f"Dataset saved to: {dataset_dir / 'flores200'}")

if __name__ == "__main__":
    prepare_flores200()
