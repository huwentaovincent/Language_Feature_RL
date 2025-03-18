import os
import argparse
import logging
from tqdm import tqdm
from datasets import load_dataset, Dataset
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("prepare_dataset")

def prepare_flores200(output_dir="assets/dataset", 
                      split="all", 
                      languages=None, 
                      force_download=False,
                      trust_remote_code=True):
    """
    Download and prepare the FLORES200 dataset.
    
    Args:
        output_dir: Directory to save the dataset
        split: Dataset split to download ("train", "dev", "devtest", "all")
        languages: List of language codes to include (None for all languages)
        force_download: If True, redownload even if files exist
        trust_remote_code: Whether to trust remote code when loading the dataset
    
    Returns:
        Path to the prepared dataset
    """
    # Create dataset directory if it doesn't exist
    dataset_dir = Path(output_dir)
    flores_dir = dataset_dir / "flores200"
    
    # Check if dataset already exists
    if flores_dir.exists() and not force_download:
        logger.info(f"Dataset already exists at {flores_dir}. Use --force to redownload.")
        return flores_dir
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine splits to download
    if split == "all":
        splits = ["train", "dev", "devtest"]
    else:
        splits = [split]
    
    logger.info(f"Downloading FLORES200 dataset (splits: {', '.join(splits)})...")
    
    try:
        # Load the dataset with trust_remote_code=True
        datasets = {}
        for s in splits:
            logger.info(f"Loading split: {s}")
            datasets[s] = load_dataset(
                "Muennighoff/flores200", 
                split=s,
                trust_remote_code=trust_remote_code,
            )
            
            # Filter languages if specified
            if languages:
                logger.info(f"Filtering to languages: {', '.join(languages)}")
                # Create a filtered dataset
                filtered_examples = []
                for example in tqdm(datasets[s], desc=f"Filtering {s} split"):
                    filtered_example = {k: v for k, v in example.items() if k in languages or k not in example.get("lang_ids", {})}
                    filtered_examples.append(filtered_example)
                
                # Create a new dataset with filtered examples
                datasets[s] = Dataset.from_dict({k: [example[k] for example in filtered_examples if k in example] 
                                               for k in filtered_examples[0].keys()})
        
        # Save the dataset locally
        logger.info(f"Saving dataset to {flores_dir}...")
        
        # Create a dictionary with all splits
        combined_dataset = {s: datasets[s] for s in splits}
        
        # Save to disk
        for s, ds in combined_dataset.items():
            save_path = flores_dir / s
            ds.save_to_disk(str(save_path))
            logger.info(f"Saved {s} split to {save_path}")
        
        # Create a metadata file with information about the dataset
        with open(flores_dir / "metadata.txt", "w") as f:
            f.write(f"Dataset: FLORES200\n")
            f.write(f"Splits: {', '.join(splits)}\n")
            if languages:
                f.write(f"Languages: {', '.join(languages)}\n")
            else:
                f.write("Languages: all\n")
            
            # Add some statistics
            for s, ds in combined_dataset.items():
                f.write(f"\n{s} split:\n")
                f.write(f"  Num examples: {len(ds)}\n")
                if len(ds) > 0:
                    f.write(f"  Features: {', '.join(ds.features.keys())}\n")
        
        logger.info("Dataset preparation completed!")
        logger.info(f"Dataset saved to: {flores_dir}")
        
        return flores_dir
    
    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        # Clean up partially downloaded files if there was an error
        if flores_dir.exists():
            logger.info(f"Cleaning up partial download at {flores_dir}")
            shutil.rmtree(flores_dir)
        raise

def main():
    parser = argparse.ArgumentParser(description="Prepare FLORES200 dataset for language feature analysis")
    parser.add_argument("--output_dir", type=str, default="assets/dataset", 
                        help="Directory to save the dataset")
    parser.add_argument("--split", type=str, default="all", choices=["train", "dev", "devtest", "all"],
                        help="Dataset split to download")
    parser.add_argument("--languages", type=str, nargs="+", 
                        default=["zh", "ja", "es", "en", "de", "fr"],
                        help="List of language codes to include (default: zh, ja, es, en, de, fr)")
    parser.add_argument("--all_languages", action="store_true",
                        help="Include all languages (overrides --languages)")
    parser.add_argument("--force", action="store_true",
                        help="Force redownload even if files exist")
    
    args = parser.parse_args()
    
    # If all languages is specified, set languages to None
    languages = None if args.all_languages else args.languages
    
    prepare_flores200(
        output_dir=args.output_dir,
        split=args.split,
        languages=languages,
        force_download=args.force
    )

if __name__ == "__main__":
    main()