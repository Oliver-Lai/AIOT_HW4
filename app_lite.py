"""
Chinglish Text Generator - Lightweight Version for Streamlit Cloud
Optimized for memory constraints.
"""

import streamlit as st
import os

# Page configuration
st.set_page_config(
    page_title="晶晶體生成器",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="collapsed"  # Start collapsed to save space
)

# Check if we should use API mode
USE_API = os.getenv('USE_API', 'false').lower() == 'true'

if USE_API:
    st.error("""
    ⚠️ **API 模式尚未實現**
    
    此應用需要本地模型，但 Streamlit Cloud 免費版記憶體不足。
    
    **建議方案：**
    1. 使用 Hugging Face Spaces（免費，資源更多）
    2. 升級到 Streamlit Cloud 付費方案
    3. 部署到其他平台（Railway, Render, Google Cloud）
    
    詳見：[MEMORY_OPTIMIZATION.md](https://github.com/YOUR_REPO/MEMORY_OPTIMIZATION.md)
    """)
    st.stop()

# Import heavy modules only when needed
try:
    from model_utils import load_model, generate_chinglish_text
    from tts_utils import synthesize_speech
except ImportError as e:
    st.error(f"導入模組失敗: {e}")
    st.stop()

def init_session_state():
    """Initialize session state variables."""
    if 'generated_text' not in st.session_state:
        st.session_state.generated_text = None
    if 'audio_path' not in st.session_state:
        st.session_state.audio_path = None

def show_memory_warning():
    """Show memory usage warning."""
    st.warning("""
    ⚠️ **Streamlit Cloud 資源限制**
    
    免費版 RAM 約 1GB，此應用可能接近限制。
    如遇錯誤，請考慮：
    - 使用 Hugging Face Spaces
    - 升級 Streamlit 方案
    - 本地部署
    """)

def main():
    """Main application function."""
    init_session_state()
    
    # Header
    st.title("🗣️ 晶晶體生成器")
    st.markdown("""
    輸入主題，生成晶晶體風格文字（中英混合）並轉換成語音。
    
    > 💡 **晶晶體**: 留學生常見的中英夾雜表達方式
    """)
    
    # Show memory warning on Streamlit Cloud
    if os.getenv('STREAMLIT_RUNTIME_ENV') == 'cloud':
        show_memory_warning()
    
    # Load model
    with st.spinner("⏳ 正在載入模型（首次需 3-5 分鐘）..."):
        try:
            model, tokenizer = load_model()
        except Exception as e:
            st.error(f"模型載入失敗: {e}")
            st.info("""
            **可能原因：記憶體不足**
            
            Streamlit Cloud 免費版限制約 1GB RAM，無法運行此模型。
            
            **解決方案：**
            1. 使用 [Hugging Face Spaces](https://huggingface.co/spaces) 部署
            2. 升級 Streamlit Cloud 方案
            3. 本地運行：`./run.sh`
            """)
            return
    
    if model is None or tokenizer is None:
        st.error("❌ 模型載入失敗")
        return
    
    st.success("✓ 模型載入成功")
    
    # Simplified settings in sidebar
    with st.sidebar:
        st.header("⚙️ 設定")
        
        max_length = st.slider(
            "文字長度",
            min_value=50,
            max_value=150,  # Reduced max
            value=100,  # Reduced default
            step=10
        )
        
        temperature = st.slider(
            "創意程度",
            min_value=0.6,
            max_value=1.2,
            value=0.8,
            step=0.1
        )
    
    # Topic input (simplified)
    st.subheader("📝 輸入主題")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        topic = st.text_input(
            "主題",
            placeholder="例如：週末計畫",
            max_chars=100  # Reduced
        )
    
    with col2:
        examples = ["週末計畫", "咖啡廳", "健身房"]
        selected = st.selectbox("快選", [""] + examples)
        if selected:
            topic = selected
    
    # Generate button
    if st.button("✨ 生成", type="primary", disabled=not topic):
        st.divider()
        
        # Text generation
        result = generate_chinglish_text(
            model=model,
            tokenizer=tokenizer,
            topic=topic,
            max_length=max_length,
            temperature=temperature,
            timeout=45  # Reduced timeout
        )
        
        if not result['success']:
            st.error(f"❌ 生成失敗: {result['error']}")
            return
        
        # Display text
        generated_text = result['text']
        st.session_state.generated_text = generated_text
        
        st.markdown("**生成的文字：**")
        st.info(generated_text)
        
        # Stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("生成時間", f"{result['generation_time']:.1f}s")
        with col2:
            st.metric("字元數", len(generated_text))
        
        # TTS (optional, can be disabled to save memory)
        if st.checkbox("🔊 生成語音", value=False):
            st.divider()
            audio_result = synthesize_speech(
                text=generated_text,
                audio_dir="temp_audio"
            )
            
            if audio_result['success']:
                st.success(f"✓ 語音合成成功")
                with open(audio_result['audio_path'], 'rb') as f:
                    st.audio(f.read(), format='audio/mp3')
            else:
                st.error(f"❌ 語音合成失敗: {audio_result['error']}")

if __name__ == "__main__":
    main()
