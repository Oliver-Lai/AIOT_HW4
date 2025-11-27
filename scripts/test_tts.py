"""
Test script for TTS (Text-to-Speech) functionality.
Tests gTTS integration with mixed Chinese-English text.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_utils import (
    synthesize_speech,
    ensure_audio_directory,
    generate_audio_filename,
    detect_language_segments,
    cleanup_old_audio_files,
    get_audio_stats
)


def test_audio_directory():
    """Test audio directory creation."""
    print("\n" + "="*60)
    print("測試 1: 音訊目錄創建")
    print("="*60)
    
    try:
        audio_path = ensure_audio_directory("temp_audio")
        print(f"✓ 音訊目錄已創建: {audio_path}")
        print(f"✓ 目錄存在: {audio_path.exists()}")
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False


def test_filename_generation():
    """Test audio filename generation."""
    print("\n" + "="*60)
    print("測試 2: 音訊檔案命名")
    print("="*60)
    
    try:
        # Test UUID-based naming
        filename1 = generate_audio_filename("test text", use_uuid=True)
        print(f"✓ UUID 命名: {filename1}")
        
        # Test timestamp-based naming
        filename2 = generate_audio_filename("test text", use_uuid=False)
        print(f"✓ 時間戳命名: {filename2}")
        
        # Verify filenames are unique
        filename3 = generate_audio_filename("test text", use_uuid=True)
        if filename1 != filename3:
            print(f"✓ 檔名唯一性驗證通過")
            return True
        else:
            print(f"✗ 檔名重複")
            return False
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False


def test_language_detection():
    """Test language segment detection."""
    print("\n" + "="*60)
    print("測試 3: 語言片段偵測")
    print("="*60)
    
    test_texts = [
        "今天天氣真好 we should go outside",
        "I love programming 我最喜歡寫程式",
        "Pure English text",
        "純中文文字",
        "這個 project 真的很 challenging 但是我覺得很 interesting"
    ]
    
    try:
        for text in test_texts:
            segments = detect_language_segments(text)
            print(f"\n文字: {text}")
            print(f"片段數: {len(segments)}")
            for seg_text, lang in segments:
                print(f"  - [{lang}] {repr(seg_text)}")
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False


def test_tts_synthesis():
    """Test TTS synthesis with various text samples."""
    print("\n" + "="*60)
    print("測試 4: TTS 語音合成")
    print("="*60)
    
    # Note: This requires mock streamlit spinner or will fail
    # We'll use a simple implementation without streamlit for testing
    test_samples = [
        ("我今天 really want to go shopping", "混合文字 (中英文)"),
        ("Let's meet at the coffee shop", "純英文"),
        ("今天是個美好的一天", "純中文"),
        ("這個 weekend 我想去 hiking 因為 weather forecast 說會很 sunny", "高密度混合")
    ]
    
    results = []
    
    for text, description in test_samples:
        print(f"\n測試樣本: {description}")
        print(f"文字: {text}")
        
        # For testing without streamlit, we need to mock the spinner
        # This is a simplified test that shows the function structure
        try:
            print(f"⚠️  注意: 此測試需要網路連線（gTTS 使用 Google API）")
            print(f"⚠️  在非 Streamlit 環境中執行，spinner 功能會被跳過")
            
            # We can't actually run this without streamlit context
            # Just validate the function exists and has correct signature
            import inspect
            sig = inspect.signature(synthesize_speech)
            params = list(sig.parameters.keys())
            
            print(f"✓ synthesize_speech 函數存在")
            print(f"✓ 參數: {params}")
            
            expected_params = ['text', 'audio_dir', 'language', 'slow']
            if all(p in params for p in expected_params):
                print(f"✓ 參數驗證通過")
                results.append(True)
            else:
                print(f"✗ 參數不完整")
                results.append(False)
                
        except Exception as e:
            print(f"✗ 失敗: {e}")
            results.append(False)
    
    return all(results)


def test_audio_stats():
    """Test audio statistics function."""
    print("\n" + "="*60)
    print("測試 5: 音訊檔案統計")
    print("="*60)
    
    try:
        stats = get_audio_stats("temp_audio")
        print(f"✓ 檔案數量: {stats['file_count']}")
        print(f"✓ 總大小: {stats['total_size_mb']:.2f} MB")
        print(f"✓ 最舊檔案年齡: {stats['oldest_file_age_minutes']:.1f} 分鐘")
        return True
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False


def test_cleanup():
    """Test audio cleanup function."""
    print("\n" + "="*60)
    print("測試 6: 音訊檔案清理")
    print("="*60)
    
    try:
        result = cleanup_old_audio_files(
            audio_dir="temp_audio",
            max_age_seconds=3600,
            keep_recent=10
        )
        
        print(f"✓ 清理成功: {result['success']}")
        print(f"✓ 刪除檔案: {result['files_deleted']}")
        print(f"✓ 保留檔案: {result['files_kept']}")
        return result['success']
    except Exception as e:
        print(f"✗ 失敗: {e}")
        return False


def main():
    """Run all TTS tests."""
    print("="*60)
    print("Chinglish Generator - TTS 模組測試")
    print("="*60)
    
    tests = [
        ("音訊目錄創建", test_audio_directory),
        ("檔案命名生成", test_filename_generation),
        ("語言片段偵測", test_language_detection),
        ("TTS 語音合成", test_tts_synthesis),
        ("音訊統計資訊", test_audio_stats),
        ("音訊檔案清理", test_cleanup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} 發生錯誤: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ 通過" if passed else "✗ 失敗"
        print(f"{status}: {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print("\n" + "="*60)
    print(f"總測試數: {total_tests}")
    print(f"通過: {passed_tests}")
    print(f"失敗: {total_tests - passed_tests}")
    print(f"通過率: {passed_tests/total_tests*100:.1f}%")
    print("="*60)
    
    # Note about actual TTS testing
    print("\n⚠️  注意事項:")
    print("- TTS 實際語音合成需要在 Streamlit 應用中測試")
    print("- gTTS 需要網路連線才能運作")
    print("- 建議在 Web 介面中進行完整的音訊品質測試")


if __name__ == "__main__":
    main()
