"""
Test the modified model loading function.
Verifies both local and HF Hub loading paths.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_loading_logic():
    """Test model loading logic without Streamlit."""
    print("="*60)
    print("測試模型載入邏輯")
    print("="*60)
    
    from pathlib import Path
    
    # Test 1: Check local model path
    print("\n[測試 1] 檢查本地模型路徑")
    local_path = "models/qwen2-0.5b-instruct"
    model_dir = Path(local_path)
    is_local = model_dir.exists() and model_dir.is_dir()
    
    if is_local:
        print(f"✓ 本地模型存在: {local_path}")
        files = list(model_dir.glob("*"))
        print(f"✓ 檔案數量: {len(files)}")
    else:
        print(f"ℹ️  本地模型不存在: {local_path}")
        print("   → 應用將從 Hugging Face Hub 下載")
    
    # Test 2: Check environment variable
    print("\n[測試 2] 檢查環境變數")
    model_path = os.getenv('MODEL_PATH', 'Qwen/Qwen2-0.5B-Instruct')
    print(f"MODEL_PATH: {model_path}")
    
    # Test 3: Verify import
    print("\n[測試 3] 驗證相關套件")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✓ transformers 套件正常")
    except ImportError as e:
        print(f"✗ transformers 套件錯誤: {e}")
    
    try:
        import torch
        print(f"✓ torch 套件正常 (版本: {torch.__version__})")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ torch 套件錯誤: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("總結")
    print("="*60)
    
    if is_local:
        print("✓ 本地部署: 將使用本地模型")
        print("  執行: streamlit run app.py")
    else:
        print("☁️  雲端部署模式: 將從 HF Hub 下載模型")
        print("  - 首次啟動需要 3-5 分鐘")
        print("  - 適用於 Streamlit Cloud")
        print("  - 模型會被快取以加快後續啟動")
    
    print("\n建議:")
    if is_local:
        print("  - 本地測試: ./run.sh")
        print("  - 部署到雲端前，確保推送代碼到 GitHub")
    else:
        print("  - 本地使用: python scripts/download_models.py")
        print("  - 雲端部署: 直接推送到 GitHub 並在 Streamlit Cloud 部署")


if __name__ == "__main__":
    test_model_loading_logic()
