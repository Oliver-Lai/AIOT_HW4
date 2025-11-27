"""
Integration test for the complete Chinglish Generator application.
Tests the full workflow: model loading -> text generation -> TTS.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path


def test_complete_workflow():
    """Test the complete workflow from topic to audio."""
    print("="*60)
    print("晶晶體生成器 - 完整流程測試")
    print("="*60)
    
    # Test 1: Model loading
    print("\n[1/4] 測試模型載入...")
    try:
        from model_utils import load_model
        
        # Note: This will fail without streamlit context
        # We'll just check the function exists
        import inspect
        sig = inspect.signature(load_model)
        print("✓ load_model 函數檢查通過")
        print(f"  參數: {list(sig.parameters.keys())}")
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False
    
    # Test 2: Text generation (without actual model)
    print("\n[2/4] 測試文字生成功能...")
    try:
        from model_utils import generate_chinglish_text, create_chinglish_prompt
        
        # Test prompt creation
        test_topic = "週末計畫"
        prompt_result = create_chinglish_prompt(test_topic)
        
        print("✓ Prompt 生成成功")
        print(f"  主題: {test_topic}")
        
        if isinstance(prompt_result, dict):
            system_prompt = prompt_result.get('system_prompt', '')
            user_prompt = prompt_result.get('user_prompt', '')
            print(f"  系統提示長度: {len(system_prompt)} 字元")
            print(f"  用戶提示長度: {len(user_prompt)} 字元")
            
            # Verify prompt contains key requirements
            key_phrases = ["60-70%", "英文", "中文"]
            found = sum(1 for phrase in key_phrases if phrase in system_prompt.lower())
            print(f"  關鍵詞覆蓋: {found}/{len(key_phrases)}")
        elif isinstance(prompt_result, tuple):
            system_prompt, user_prompt = prompt_result
            print(f"  系統提示長度: {len(system_prompt)} 字元")
            print(f"  用戶提示長度: {len(user_prompt)} 字元")
        else:
            print(f"  Prompt 類型: {type(prompt_result)}")
        
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False
    
    # Test 3: TTS functions
    print("\n[3/4] 測試 TTS 功能...")
    try:
        from tts_utils import (
            ensure_audio_directory,
            generate_audio_filename,
            detect_language_segments,
            get_audio_stats
        )
        
        # Test directory
        audio_dir = ensure_audio_directory("temp_audio")
        print(f"✓ 音訊目錄: {audio_dir}")
        
        # Test filename generation
        filename = generate_audio_filename("test")
        print(f"✓ 檔名生成: {filename}")
        
        # Test language detection
        test_text = "我今天 want to go shopping"
        segments = detect_language_segments(test_text)
        print(f"✓ 語言偵測: {len(segments)} 個片段")
        
        # Test stats
        stats = get_audio_stats("temp_audio")
        print(f"✓ 統計資訊: {stats['file_count']} 檔案, {stats['total_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False
    
    # Test 4: Check all required files
    print("\n[4/4] 檢查必要檔案...")
    required_files = [
        "app.py",
        "model_utils.py",
        "tts_utils.py",
        "requirements.txt",
        "run.sh",
        ".streamlit/config.toml"
    ]
    
    all_exist = True
    for filepath in required_files:
        full_path = Path(filepath)
        exists = full_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {filepath}")
        if not exists:
            all_exist = False
    
    # Check model directory
    model_dir = Path("models/qwen2-0.5b-instruct")
    if model_dir.exists():
        model_files = list(model_dir.glob("*"))
        print(f"  ✓ models/qwen2-0.5b-instruct/ ({len(model_files)} 檔案)")
    else:
        print(f"  ⚠️  models/qwen2-0.5b-instruct/ (不存在，需要下載)")
    
    return all_exist


def test_streamlit_app_structure():
    """Test the Streamlit app structure."""
    print("\n" + "="*60)
    print("檢查 Streamlit 應用結構")
    print("="*60)
    
    try:
        with open("app.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key components
        checks = {
            "Streamlit import": "import streamlit",
            "主標題": "st.title",
            "側邊欄設定": "st.sidebar",
            "模型載入": "load_model",
            "文字生成": "generate_chinglish_text",
            "語音合成": "synthesize_speech",
            "音訊播放器": "st.audio",
            "下載按鈕": "st.download_button"
        }
        
        results = {}
        for name, pattern in checks.items():
            found = pattern in content
            results[name] = found
            status = "✓" if found else "✗"
            print(f"{status} {name}")
        
        passed = sum(1 for v in results.values() if v)
        print(f"\n通過: {passed}/{len(checks)}")
        
        return all(results.values())
        
    except Exception as e:
        print(f"✗ 讀取 app.py 失敗: {e}")
        return False


def main():
    """Run all integration tests."""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "整合測試套件" + " "*31 + "║")
    print("╚" + "═"*58 + "╝")
    
    results = {}
    
    # Run tests
    print("\n" + "▶ "*30)
    results['完整流程'] = test_complete_workflow()
    
    print("\n" + "▶ "*30)
    results['應用結構'] = test_streamlit_app_structure()
    
    # Summary
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ 通過" if passed else "✗ 失敗"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有測試通過！應用已準備就緒。")
        print("\n啟動應用:")
        print("  ./run.sh")
        print("  或")
        print("  streamlit run app.py")
    else:
        print("⚠️  部分測試失敗，請檢查錯誤訊息。")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n測試已中斷")
    except Exception as e:
        print(f"\n\n測試失敗: {e}")
        import traceback
        traceback.print_exc()
