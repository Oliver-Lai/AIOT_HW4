"""
Chinglish Text Generator - Text-to-Speech Module
Handles audio synthesis for mixed Chinese-English text.
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
import time
from gtts import gTTS
import streamlit as st


def ensure_audio_directory(audio_dir: str = "temp_audio") -> Path:
    """
    Ensure audio output directory exists.
    
    Args:
        audio_dir: Directory path for audio files
        
    Returns:
        Path object for the audio directory
    """
    audio_path = Path(audio_dir)
    audio_path.mkdir(exist_ok=True)
    return audio_path


def generate_audio_filename(text: str, use_uuid: bool = True) -> str:
    """
    Generate unique filename for audio file.
    
    Args:
        text: Text content (used for hash if not using UUID)
        use_uuid: If True, use UUID; if False, use timestamp
        
    Returns:
        Filename string (without directory path)
    """
    if use_uuid:
        # Use UUID for guaranteed uniqueness
        unique_id = str(uuid.uuid4())[:8]
    else:
        # Use timestamp
        unique_id = str(int(time.time() * 1000))
    
    return f"chinglish_{unique_id}.mp3"


def detect_language_segments(text: str) -> list:
    """
    Detect Chinese and English segments in text.
    
    Args:
        text: Mixed language text
        
    Returns:
        List of tuples (text_segment, language)
        where language is 'zh' for Chinese or 'en' for English
    """
    segments = []
    current_segment = ""
    current_lang = None
    
    for char in text:
        # Detect if character is Chinese
        is_chinese = '\u4e00' <= char <= '\u9fff'
        # Detect if character is English letter
        is_english = char.isalpha() and ord(char) < 128
        
        if is_chinese:
            if current_lang == 'zh':
                current_segment += char
            else:
                if current_segment:
                    segments.append((current_segment, current_lang))
                current_segment = char
                current_lang = 'zh'
        elif is_english:
            if current_lang == 'en':
                current_segment += char
            else:
                if current_segment:
                    segments.append((current_segment, current_lang))
                current_segment = char
                current_lang = 'en'
        else:
            # Space, punctuation, etc - append to current segment
            current_segment += char
    
    # Add final segment
    if current_segment:
        segments.append((current_segment, current_lang))
    
    return segments


def synthesize_speech(
    text: str,
    audio_dir: str = "temp_audio",
    language: str = "zh",
    slow: bool = False
) -> Dict[str, Any]:
    """
    Synthesize speech from text using gTTS.
    
    Args:
        text: Text to synthesize (can be mixed Chinese-English)
        audio_dir: Directory to save audio file
        language: Primary language ('zh' for Chinese, 'en' for English)
        slow: If True, generate slower speech
        
    Returns:
        Dictionary with 'success', 'audio_path', 'synthesis_time', and 'error' keys
    """
    if not text or not text.strip():
        return {
            'success': False,
            'audio_path': None,
            'synthesis_time': 0,
            'error': '文字內容為空'
        }
    
    try:
        start_time = time.time()
        
        # Ensure audio directory exists
        audio_path = ensure_audio_directory(audio_dir)
        
        # Generate unique filename
        filename = generate_audio_filename(text)
        file_path = audio_path / filename
        
        # Detect if text has both languages
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        has_english = any(char.isalpha() and ord(char) < 128 for char in text)
        
        # For mixed text, use Chinese as primary (it can handle both better)
        if has_chinese and has_english:
            # Use Chinese TTS which handles mixed text relatively well
            tts_lang = 'zh-TW'  # Taiwan Mandarin for more natural pronunciation
        elif has_english and not has_chinese:
            tts_lang = 'en'
        else:
            tts_lang = 'zh-TW'
        
        # Generate speech with gTTS
        with st.spinner('🔊 正在生成語音...'):
            tts = gTTS(text=text, lang=tts_lang, slow=slow)
            tts.save(str(file_path))
        
        synthesis_time = time.time() - start_time
        
        # Verify file was created
        if not file_path.exists() or file_path.stat().st_size == 0:
            return {
                'success': False,
                'audio_path': None,
                'synthesis_time': synthesis_time,
                'error': '音訊檔案生成失敗或為空'
            }
        
        return {
            'success': True,
            'audio_path': str(file_path),
            'synthesis_time': synthesis_time,
            'error': None
        }
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error messages
        if 'connection' in error_msg.lower() or 'network' in error_msg.lower():
            error_msg = '網路連線失敗，gTTS 需要網路連線。請檢查網路狀態。'
        elif 'timeout' in error_msg.lower():
            error_msg = 'gTTS 請求超時，請稍後再試。'
        
        return {
            'success': False,
            'audio_path': None,
            'synthesis_time': 0,
            'error': f'語音合成失敗: {error_msg}'
        }


def cleanup_old_audio_files(
    audio_dir: str = "temp_audio",
    max_age_seconds: int = 3600,
    keep_recent: int = 10
) -> Dict[str, Any]:
    """
    Clean up old audio files to save disk space.
    
    Args:
        audio_dir: Directory containing audio files
        max_age_seconds: Maximum age in seconds (default: 1 hour)
        keep_recent: Number of recent files to keep regardless of age
        
    Returns:
        Dictionary with cleanup statistics
    """
    try:
        audio_path = Path(audio_dir)
        
        if not audio_path.exists():
            return {
                'success': True,
                'files_deleted': 0,
                'files_kept': 0,
                'error': None
            }
        
        # Get all MP3 files with their modification times
        audio_files = []
        for file in audio_path.glob("*.mp3"):
            mtime = file.stat().st_mtime
            audio_files.append((file, mtime))
        
        # Sort by modification time (newest first)
        audio_files.sort(key=lambda x: x[1], reverse=True)
        
        current_time = time.time()
        files_deleted = 0
        files_kept = 0
        
        for i, (file, mtime) in enumerate(audio_files):
            # Keep the most recent files
            if i < keep_recent:
                files_kept += 1
                continue
            
            # Delete old files
            age = current_time - mtime
            if age > max_age_seconds:
                file.unlink()
                files_deleted += 1
            else:
                files_kept += 1
        
        return {
            'success': True,
            'files_deleted': files_deleted,
            'files_kept': files_kept,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'files_deleted': 0,
            'files_kept': 0,
            'error': str(e)
        }


def get_audio_stats(audio_dir: str = "temp_audio") -> Dict[str, Any]:
    """
    Get statistics about audio files in the directory.
    
    Args:
        audio_dir: Directory containing audio files
        
    Returns:
        Dictionary with file count and total size
    """
    try:
        audio_path = Path(audio_dir)
        
        if not audio_path.exists():
            return {
                'file_count': 0,
                'total_size_mb': 0,
                'oldest_file_age_minutes': 0
            }
        
        audio_files = list(audio_path.glob("*.mp3"))
        total_size = sum(f.stat().st_size for f in audio_files)
        
        # Get oldest file age
        oldest_age_minutes = 0
        if audio_files:
            oldest_mtime = min(f.stat().st_mtime for f in audio_files)
            oldest_age_minutes = (time.time() - oldest_mtime) / 60
        
        return {
            'file_count': len(audio_files),
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_file_age_minutes': oldest_age_minutes
        }
        
    except Exception as e:
        return {
            'file_count': 0,
            'total_size_mb': 0,
            'oldest_file_age_minutes': 0,
            'error': str(e)
        }
