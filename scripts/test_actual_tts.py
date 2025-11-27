"""
Test actual TTS synthesis with gTTS (requires internet connection).
This script creates actual audio files for quality testing.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gtts import gTTS
from pathlib import Path
import time


def test_actual_synthesis():
    """Test actual TTS synthesis without Streamlit dependency."""
    print("="*60)
    print("實際 TTS 語音合成測試")
    print("⚠️  需要網路連線")
    print("="*60)
    
    # Ensure audio directory exists
    audio_dir = Path("temp_audio")
    audio_dir.mkdir(exist_ok=True)
    
    test_samples = [
        {
            'text': '我今天 really want to go to the beach 因為 weather 超好',
            'filename': 'test_mixed_1.mp3',
            'description': '混合文字測試 1'
        },
        {
            'text': 'This weekend I want to go hiking 然後 maybe have some barbecue',
            'filename': 'test_mixed_2.mp3',
            'description': '混合文字測試 2'
        },
        {
            'text': '我覺得這個 project 真的很 challenging 但是 very interesting',
            'filename': 'test_mixed_3.mp3',
            'description': '高密度混合測試'
        },
        {
            'text': 'Let me tell you about my day',
            'filename': 'test_english.mp3',
            'description': '純英文測試'
        },
        {
            'text': '今天是個美好的一天',
            'filename': 'test_chinese.mp3',
            'description': '純中文測試'
        }
    ]
    
    results = []
    
    for i, sample in enumerate(test_samples, 1):
        print(f"\n[{i}/{len(test_samples)}] {sample['description']}")
        print(f"文字: {sample['text']}")
        
        try:
            start_time = time.time()
            
            # Detect language
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in sample['text'])
            has_english = any(char.isalpha() and ord(char) < 128 for char in sample['text'])
            
            if has_chinese and has_english:
                lang = 'zh-TW'
                lang_desc = '混合 (使用中文語音)'
            elif has_english and not has_chinese:
                lang = 'en'
                lang_desc = '英文'
            else:
                lang = 'zh-TW'
                lang_desc = '中文'
            
            print(f"語言偵測: {lang_desc}")
            
            # Generate speech
            file_path = audio_dir / sample['filename']
            tts = gTTS(text=sample['text'], lang=lang, slow=False)
            tts.save(str(file_path))
            
            synthesis_time = time.time() - start_time
            file_size = file_path.stat().st_size / 1024  # KB
            
            print(f"✓ 合成成功")
            print(f"  - 耗時: {synthesis_time:.2f} 秒")
            print(f"  - 檔案: {file_path}")
            print(f"  - 大小: {file_size:.1f} KB")
            
            results.append({
                'success': True,
                'description': sample['description'],
                'time': synthesis_time,
                'size': file_size
            })
            
        except Exception as e:
            print(f"✗ 失敗: {e}")
            results.append({
                'success': False,
                'description': sample['description'],
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("測試總結")
    print("="*60)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\n成功: {len(successful)}/{len(results)}")
    
    if successful:
        avg_time = sum(r['time'] for r in successful) / len(successful)
        total_size = sum(r['size'] for r in successful)
        print(f"平均合成時間: {avg_time:.2f} 秒")
        print(f"總檔案大小: {total_size:.1f} KB")
    
    if failed:
        print(f"\n失敗: {len(failed)}")
        for r in failed:
            print(f"  - {r['description']}: {r.get('error', 'Unknown error')}")
    
    # List generated files
    print("\n" + "="*60)
    print("生成的音訊檔案")
    print("="*60)
    
    audio_files = sorted(audio_dir.glob("test_*.mp3"))
    if audio_files:
        for f in audio_files:
            size = f.stat().st_size / 1024
            print(f"  - {f.name} ({size:.1f} KB)")
        print(f"\n可以使用音訊播放器測試這些檔案的品質")
    else:
        print("  (無檔案)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        test_actual_synthesis()
    except KeyboardInterrupt:
        print("\n\n測試已中斷")
    except Exception as e:
        print(f"\n\n測試失敗: {e}")
