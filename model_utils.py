"""
Chinglish Text Generator - Model Integration Module
Handles model loading, text generation with Chinglish-style prompts.
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Dict, Any
import time
import os


@st.cache_resource
def load_model(model_path: Optional[str] = None):
    """
    Load Qwen2 model and tokenizer.
    Supports both local path and Hugging Face Hub.
    Uses Streamlit cache to avoid reloading on every rerun.
    
    Args:
        model_path: Path to local model directory or HF model ID.
                   If None, uses environment variable MODEL_PATH or defaults to HF Hub.
        
    Returns:
        Tuple of (model, tokenizer) or (None, None) if loading fails
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Determine model source
        if model_path is None:
            # Check environment variable first (for Streamlit Cloud secrets)
            model_path = os.getenv('MODEL_PATH', 'Qwen/Qwen2-0.5B-Instruct')
        
        # Check if it's a local path
        model_dir = Path(model_path)
        is_local = model_dir.exists() and model_dir.is_dir()
        
        if is_local:
            st.info(f"📁 從本地載入模型: {model_path}")
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                local_files_only=True
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                local_files_only=True,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Download from Hugging Face Hub
            st.info(f"☁️ 從 Hugging Face Hub 下載模型: {model_path}")
            st.warning("⏳ 首次下載需要 3-5 分鐘，請耐心等候...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model with optimizations for cloud deployment
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True  # Important for Streamlit Cloud
            )
            
            st.success("✓ 模型下載並載入成功！")
        
        return model, tokenizer
        
    except Exception as e:
        st.error(f"❌ 模型載入失敗: {str(e)}")
        st.info("""
        **解決方案：**
        - 本地部署：執行 `python scripts/download_models.py` 下載模型
        - Streamlit Cloud：模型會自動從 Hugging Face Hub 下載
        - 如遇到記憶體問題，請聯絡管理員
        """)
        return None, None


def create_chinglish_prompt(topic: str) -> str:
    """
    Create a system prompt that guides the model to generate Chinglish-style text.
    
    Args:
        topic: User-provided topic
        
    Returns:
        Formatted prompt string
    """
    system_prompt = """你是一個在美國留學多年的台灣學生，說話習慣大量使用英文，只有在必要時才用中文連接句子（晶晶體風格）。

重要規則：
- 英文比例要高（60-70%），中文主要用於連接詞和簡單表達
- 所有專業術語、動詞、形容詞優先用英文
- 名詞、概念、活動都用英文表達
- 只用中文說：這個、那個、我、你、的、了、嗎、吧、啊、就、也、還、但是、所以
- 完全自然的 code-switching，像真正的留學生一樣

正確範例（英文含量高）：
- "我 yesterday 的 presentation 真的 super stressful，professor 一直 keep asking 一些 tricky questions，honestly 我 almost couldn't handle it。"
- "這個 weekend 我想 go hiking with some friends，but 要 first finish 我的 programming assignment，因為 deadline 是 Monday morning。"
- "今天 lecture 上到一半我的 laptop suddenly crashed，整個 very embarrassing，同學們 all looked at me，我 just wanted to leave the classroom immediately。"
- "最近 midterm season 真的 too intense，每天要 study 到 very late，sometimes 連 sleep 都 not enough，感覺 super exhausted。"

錯誤範例（中文太多，要避免）：
- "這個作業真的很難，我不知道怎麼做。" ❌ 太多中文
- "今天很累。" ❌ 沒有英文
"""
    
    user_prompt = f"""請根據以下主題生成一段對話或描述。記住：要大量使用英文（60-70%），中文只用於簡單連接。

主題：{topic}

請生成 100-150 字，展現真實留學生說話風格："""
    
    full_prompt = f"{system_prompt}\n{user_prompt}"
    return full_prompt


def generate_chinglish_text(
    model,
    tokenizer,
    topic: str,
    max_length: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Generate Chinglish-style text based on user topic.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        topic: User-provided topic
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling parameter
        timeout: Maximum time in seconds for generation (default: 60)
        
    Returns:
        Dictionary with 'text', 'generation_time', 'success', and 'error' keys
    """
    if model is None or tokenizer is None:
        return {
            'success': False,
            'text': '',
            'generation_time': 0,
            'error': '模型未載入'
        }
    
    # Validate input
    if not topic or not topic.strip():
        return {
            'success': False,
            'text': '',
            'generation_time': 0,
            'error': '主題不能為空'
        }
    
    if len(topic) > 200:
        return {
            'success': False,
            'text': '',
            'generation_time': 0,
            'error': '主題長度不能超過 200 字元'
        }
    
    try:
        start_time = time.time()
        
        # Create prompt
        prompt = create_chinglish_prompt(topic.strip())
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate with timeout handling
        with st.spinner('🤖 正在生成晶晶體文字...'):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Check for timeout
        if generation_time > timeout:
            return {
                'success': False,
                'text': '',
                'generation_time': generation_time,
                'error': f'生成超時（超過 {timeout} 秒）'
            }
        
        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after the prompt)
        generated_text = full_text[len(prompt):].strip()
        
        # Remove any remaining prompt fragments
        if "主題：" in generated_text:
            parts = generated_text.split("主題：", 1)
            if len(parts) > 1:
                generated_text = parts[1].strip()
        
        # Validate output contains both languages
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in generated_text)
        has_english = any(char.isalpha() and ord(char) < 128 for char in generated_text)
        
        # Retry once if validation fails and we haven't timed out
        if not (has_chinese and has_english) and (time.time() - start_time) < timeout:
            st.warning('⚠️ 生成的文字語言不夠混合，正在重試...')
            
            retry_start = time.time()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=min(temperature + 0.15, 1.0),  # Increase randomness
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            if (time.time() - retry_start) > timeout:
                return {
                    'success': False,
                    'text': generated_text,  # Return original attempt
                    'generation_time': time.time() - start_time,
                    'error': '重試超時'
                }
            
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = full_text[len(prompt):].strip()
            
            if "主題：" in generated_text:
                parts = generated_text.split("主題：", 1)
                if len(parts) > 1:
                    generated_text = parts[1].strip()
        
        generation_time = time.time() - start_time
        
        # Final validation
        if not generated_text or len(generated_text) < 20:
            return {
                'success': False,
                'text': generated_text,
                'generation_time': generation_time,
                'error': '生成的文字太短或為空'
            }
        
        return {
            'success': True,
            'text': generated_text,
            'generation_time': generation_time,
            'error': None
        }
        
    except KeyboardInterrupt:
        return {
            'success': False,
            'text': '',
            'generation_time': 0,
            'error': '生成被使用者中斷'
        }
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            return {
                'success': False,
                'text': '',
                'generation_time': 0,
                'error': '記憶體不足，請嘗試使用較小的模型或減少 max_length'
            }
        return {
            'success': False,
            'text': '',
            'generation_time': 0,
            'error': f'執行錯誤: {str(e)}'
        }
    except Exception as e:
        return {
            'success': False,
            'text': '',
            'generation_time': 0,
            'error': f'未預期的錯誤: {str(e)}'
        }


def test_generation_quality(model, tokenizer, test_topics: list):
    """
    Test text generation quality with various topics.
    
    Args:
        model: Loaded language model
        tokenizer: Loaded tokenizer
        test_topics: List of test topics
    """
    st.subheader("🧪 生成品質測試")
    
    for topic in test_topics:
        st.write(f"**測試主題**: {topic}")
        
        result = generate_chinglish_text(model, tokenizer, topic, max_length=150)
        
        if result['success']:
            st.success(f"✓ 生成成功 ({result['generation_time']:.2f}秒)")
            st.write(result['text'])
            
            # Check language mixing
            text = result['text']
            has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
            has_english = any(char.isalpha() and ord(char) < 128 for char in text)
            
            if has_chinese and has_english:
                st.info("✓ 包含中英文混合")
            else:
                st.warning("⚠ 語言混合不足")
        else:
            st.error(f"✗ 生成失敗: {result['error']}")
        
        st.divider()
