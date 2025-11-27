"""
Chinglish Text Generator - Streamlit Web Application
Generate Chinglish-style text and convert to speech.
"""

import streamlit as st
import time
from pathlib import Path

# Import local modules
from model_utils import load_model, generate_chinglish_text
from tts_utils import synthesize_speech, get_audio_stats, cleanup_old_audio_files


# Page configuration
st.set_page_config(
    page_title="晶晶體生成器",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'generated_text' not in st.session_state:
        st.session_state.generated_text = None
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0


def main():
    """Main application function."""
    init_session_state()
    
    # Header
    st.title("🗣️ 晶晶體生成器")
    st.markdown("""
    輸入一個主題，AI 會生成具有「晶晶體」風格的文字 (中英文混合)，並轉換成語音！
    
    > 💡 **晶晶體**: 留學生或海外工作者講中文時常見的中英夾雜表達方式
    """)
    
    # Sidebar - Settings & Info
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # Model settings
        st.subheader("文字生成設定")
        max_length = st.slider(
            "最大長度",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            help="生成文字的最大長度（字元數）"
        )
        
        temperature = st.slider(
            "創意程度",
            min_value=0.5,
            max_value=1.5,
            value=0.8,
            step=0.1,
            help="較高的值會產生更有創意但可能不太連貫的文字"
        )
        
        top_p = st.slider(
            "多樣性",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="控制文字的多樣性"
        )
        
        # TTS settings
        st.subheader("語音合成設定")
        slow_speech = st.checkbox(
            "慢速語音",
            value=False,
            help="生成較慢的語音（更清晰）"
        )
        
        # Audio cleanup
        st.subheader("音訊管理")
        audio_stats = get_audio_stats()
        st.metric("暫存音訊檔案", f"{audio_stats['file_count']} 個")
        st.metric("佔用空間", f"{audio_stats['total_size_mb']:.2f} MB")
        
        if st.button("🗑️ 清理舊音訊", help="刪除超過 1 小時的音訊檔案"):
            result = cleanup_old_audio_files(max_age_seconds=3600, keep_recent=5)
            if result['success']:
                st.success(f"✓ 已刪除 {result['files_deleted']} 個舊檔案")
            else:
                st.error(f"清理失敗: {result['error']}")
        
        # Info
        st.divider()
        st.subheader("ℹ️ 關於")
        st.markdown("""
        **技術棧:**
        - 🤖 Qwen2-0.5B-Instruct
        - 🎤 Google TTS (gTTS)
        - 🌐 Streamlit
        
        **版本:** 1.0.0
        """)
    
    # Main content
    st.divider()
    
    # Load model
    with st.spinner("⏳ 正在載入語言模型..."):
        model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        st.error("""
        ❌ **模型載入失敗**
        
        請確保已下載模型檔案。執行以下命令：
        ```bash
        python scripts/download_models.py
        ```
        """)
        return
    
    st.success("✓ 語言模型載入成功")
    
    # Topic input
    st.subheader("📝 輸入主題")
    
    # Example topics
    example_topics = [
        "週末計畫",
        "找工作面試",
        "咖啡廳閒聊",
        "健身房",
        "追劇心得",
        "旅遊規劃"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic = st.text_input(
            "你想聊什麼？",
            placeholder="例如：週末計畫、咖啡廳閒聊、健身房...",
            max_chars=200,
            help="輸入任何你想要生成晶晶體文字的主題"
        )
    
    with col2:
        st.markdown("**快速選擇:**")
        for example in example_topics[:3]:
            if st.button(example, key=f"example_{example}"):
                topic = example
                st.rerun()
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        generate_button = st.button(
            "✨ 生成晶晶體",
            type="primary",
            use_container_width=True,
            disabled=not topic or len(topic.strip()) == 0
        )
    
    # Generation process
    if generate_button and topic:
        st.divider()
        
        # Text generation
        st.subheader("💬 生成結果")
        
        with st.spinner("🤔 AI 正在思考..."):
            result = generate_chinglish_text(
                model=model,
                tokenizer=tokenizer,
                topic=topic,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                timeout=60
            )
        
        if not result['success']:
            st.error(f"❌ 生成失敗: {result['error']}")
            return
        
        # Display generated text
        generated_text = result['text']
        st.session_state.generated_text = generated_text
        st.session_state.generation_count += 1
        
        # Show text in a nice box
        st.markdown("**生成的晶晶體文字:**")
        st.info(generated_text)
        
        # Show stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("字元數", len(generated_text))
        with col2:
            st.metric("生成時間", f"{result['generation_time']:.2f}s")
        with col3:
            # Count English ratio
            english_chars = sum(1 for c in generated_text if c.isalpha() and ord(c) < 128)
            chinese_chars = sum(1 for c in generated_text if '\u4e00' <= c <= '\u9fff')
            total_chars = english_chars + chinese_chars
            english_ratio = (english_chars / total_chars * 100) if total_chars > 0 else 0
            st.metric("英文比例", f"{english_ratio:.1f}%")
        
        # Speech synthesis
        st.divider()
        st.subheader("🔊 語音合成")
        
        audio_result = synthesize_speech(
            text=generated_text,
            audio_dir="temp_audio",
            language="zh",
            slow=slow_speech
        )
        
        if audio_result['success']:
            st.session_state.audio_path = audio_result['audio_path']
            
            st.success(f"✓ 語音合成成功 (耗時 {audio_result['synthesis_time']:.2f}s)")
            
            # Audio player
            with open(audio_result['audio_path'], 'rb') as audio_file:
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
            
            # Download button
            st.download_button(
                label="📥 下載音訊檔案",
                data=audio_bytes,
                file_name=f"chinglish_{st.session_state.generation_count}.mp3",
                mime="audio/mp3"
            )
        else:
            st.error(f"❌ 語音合成失敗: {audio_result['error']}")
            st.info("💡 你仍然可以看到生成的文字內容")
    
    # Show previous results if any
    elif st.session_state.generated_text:
        st.divider()
        st.subheader("📋 上次生成的結果")
        
        with st.expander("查看上次的內容", expanded=False):
            st.info(st.session_state.generated_text)
            
            if st.session_state.audio_path and Path(st.session_state.audio_path).exists():
                with open(st.session_state.audio_path, 'rb') as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        Made with ❤️ using Streamlit | 
        <a href='https://github.com' target='_blank'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
