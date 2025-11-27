#!/usr/bin/env python3
"""
Model Download Script
Downloads Qwen2-1.5B-Instruct model to local directory for offline use.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed. Please run:")
    print("  pip install huggingface_hub")
    sys.exit(1)


def download_qwen_model(model_size="1.5B"):
    """
    Download Qwen2 model to local directory.
    
    Args:
        model_size: Model size - "0.5B" or "1.5B" (default)
    """
    # Determine model name and local directory
    if model_size == "0.5B":
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        local_dir = "models/qwen2-0.5b-instruct"
    elif model_size == "1.5B":
        model_name = "Qwen/Qwen2-1.5B-Instruct"
        local_dir = "models/qwen2-1.5b-instruct"
    else:
        print(f"Error: Unsupported model size '{model_size}'. Use '0.5B' or '1.5B'.")
        sys.exit(1)
    
    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    full_local_dir = project_root / local_dir
    
    print(f"Downloading {model_name}...")
    print(f"Target directory: {full_local_dir}")
    print("This may take several minutes depending on your internet connection.")
    print()
    
    try:
        # Download model
        snapshot_download(
            repo_id=model_name,
            local_dir=str(full_local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        print()
        print(f"✓ Successfully downloaded {model_name}")
        print(f"✓ Model saved to: {full_local_dir}")
        print()
        print("You can now run the application with:")
        print("  streamlit run app.py")
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        sys.exit(1)


def main():
    """Main function to handle command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Qwen2 model for Chinglish text generation"
    )
    parser.add_argument(
        "--size",
        choices=["0.5B", "1.5B"],
        default="1.5B",
        help="Model size to download (default: 1.5B)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Qwen2 Model Downloader")
    print("=" * 60)
    print()
    
    download_qwen_model(args.size)


if __name__ == "__main__":
    main()
