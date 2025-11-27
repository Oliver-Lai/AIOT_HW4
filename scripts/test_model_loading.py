#!/usr/bin/env python3
"""
Test script to verify model loading from local directory.
"""

import sys
from pathlib import Path

def test_model_loading():
    """Test loading Qwen2 model from local directory."""
    
    print("Testing model loading from local directory...")
    print("=" * 60)
    
    # Check if model directory exists
    model_dir = Path("models/qwen2-0.5b-instruct")
    if not model_dir.exists():
        print(f"✗ Error: Model directory not found: {model_dir}")
        print("Please run: python scripts/download_models.py --size 0.5B")
        return False
    
    print(f"✓ Model directory exists: {model_dir}")
    
    # Try to import required libraries
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ Transformers library imported successfully")
    except ImportError as e:
        print(f"✗ Error importing transformers: {e}")
        print("Please run: pip install transformers torch")
        return False
    
    # Try to load tokenizer
    try:
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"✗ Error loading tokenizer: {e}")
        return False
    
    # Try to load model
    try:
        print("\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {model.__class__.__name__}")
        print(f"  Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Try a simple generation test
    try:
        print("\nTesting text generation...")
        test_prompt = "你好，"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ Generation test successful")
        print(f"  Input: {test_prompt}")
        print(f"  Output: {generated_text}")
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Model is ready to use.")
    return True


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
