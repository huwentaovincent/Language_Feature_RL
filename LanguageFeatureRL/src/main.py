# src/main.py
import argparse
from pathlib import Path

from data.prepare_dataset import prepare_flores200
from models.prepare_model import prepare_model
from features.get_lang_features import get_language_features
from features.filter_shared_features import filter_shared_features
from training.ppo_sft import PPOTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Language Model Feature Analysis')
    parser.add_argument('--stage', type=str, choices=['all', 'prepare', 'extract', 'filter', 'train'], 
                        default='all', help='Pipeline stage to run')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Configuration file path')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.stage in ['all', 'prepare']:
        print("=== Preparing Dataset and Model ===")
        prepare_flores200()
        prepare_model()
    
    if args.stage in ['all', 'extract']:
        print("=== Extracting Language Features ===")
        get_language_features()
    
    if args.stage in ['all', 'filter']:
        print("=== Filtering Shared Features ===")
        filter_shared_features()
    
    if args.stage in ['all', 'train']:
        print("=== Training Model with PPO ===")
        trainer = PPOTrainer(
            model_path=config['model_path'],
            ja_features_path=config['features_path'],
            learning_rate=config['learning_rate']
        )
        trainer.train()

if __name__ == "__main__":
    main()