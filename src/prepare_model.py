import os
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def prepare_model():
    # Create assets directory if it doesn't exist
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Set device
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Create model directory
    model_dir = assets_dir / "gpt2-small"
    model_dir.mkdir(exist_ok=True)
    
    print("Downloading GPT-2 small model and tokenizer...")
    
    # Download model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Save model and tokenizer locally
    print(f"Saving model and tokenizer to {model_dir}")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Test the model
    print("\nTesting the model...")
    test_text = "Hello, this is a test."
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    
    print("Model test successful!")
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Output shape: {outputs.logits.shape}")
    
    print(f"\nModel and tokenizer saved to {model_dir}")
    print("You can now run get_lang_features.py")

if __name__ == "__main__":
    prepare_model()