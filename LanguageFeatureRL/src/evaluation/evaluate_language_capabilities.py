# src/evaluation/evaluate_language_capabilities.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import json
import numpy as np
from tqdm import tqdm

def evaluate_perplexity(model, tokenizer, dataset, languages):
    """Evaluate perplexity on texts of different languages"""
    results = {}
    
    for lang in languages:
        texts = [item[lang] for item in dataset["dev"][:50]]
        total_loss = 0
        total_tokens = 0
        
        for text in tqdm(texts, desc=f"Evaluating {lang}"):
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
        
        perplexity = np.exp(total_loss / total_tokens)
        results[lang] = perplexity
    
    return results

def evaluate_feature_activation(model, tokenizer, sae, dataset, feature_indices, languages):
    """Evaluate how strongly language-specific features activate on different languages"""
    # Implementation here
    
def main():
    # Load model, dataset, and features
    model_path = "assets/ppo_model"
    base_model_path = "assets/gpt2-small"
    
    # Compare fine-tuned vs base model
    results = {
        "perplexity": {
            "base_model": {},
            "fine_tuned": {}
        },
        "feature_activation": {
            "base_model": {},
            "fine_tuned": {}
        }
    }
    
    # Run evaluations
    
    # Save results
    with open("assets/results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()