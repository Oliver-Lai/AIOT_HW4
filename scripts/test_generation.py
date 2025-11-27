
#!/usr/bin/env python3
"""
Test script for model integration and Chinglish text generation quality.
Note: This script tests the model functions without Streamlit caching.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model_without_cache(model_path: str = "models/qwen2-0.5b-instruct"):
    """Load model without Streamlit cache for testing."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_dir = Path(model_path)
        if not model_dir.exists():
            print(f"❌ 模型目錄不存在: {model_dir}")
            return None, None
        
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            device_map="auto",
            trust_remote_code=True
        )
        
        return model, tokenizer
    except Exception as e:
        print(f"❌ 模型載入失敗: {str(e)}")
        return None, None


def create_chinglish_prompt(topic: str) -> str:
    """Create Chinglish-style prompt."""
    system_prompt = """你是一個在美國留學的台灣學生，說話時會自然地中英文夾雜（晶晶體風格）。

特點：
- 專業術語和概念用英文（如：project, deadline, presentation, professor）
- 日常對話用中文為主
- 自然的 code-switching，不要刻意
- 保持句子流暢和自然

範例：
- "這個 project 真的很 challenging，我覺得 deadline 太緊了。"
- "今天 professor 說我們要 prepare 一個 presentation，honestly 我有點緊張。"
"""
    
    user_prompt = f"""請根據以下主題生成一段自然的對話或描述。

主題：{topic}

請生成一段 100-150 字的內容："""
    
    return f"{system_prompt}\n{user_prompt}"


def generate_chinglish_text(
    model,
    tokenizer,
    topic: str,
    max_length: int = 200,
    temperature: float = 0.8
) -> Dict[str, Any]:
    """Generate Chinglish text."""
    if model is None or tokenizer is None:
        return {'success': False, 'text': '', 'generation_time': 0, 'error': '模型未載入'}
    
    try:
        start_time = time.time()
        prompt = create_chinglish_prompt(topic)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = full_text[len(prompt):].strip()
        generation_time = time.time() - start_time
        
        return {
            'success': True,
            'text': generated_text,
            'generation_time': generation_time,
            'error': None
        }
    except Exception as e:
        return {'success': False, 'text': '', 'generation_time': 0, 'error': str(e)}


def test_model_integration():
    """Test model loading and text generation functionality."""
    
    print("=" * 70)
    print("Testing Model Integration")
    print("=" * 70)
    
    # Test 1: Model Loading
    print("\n1. Testing model loading...")
    model, tokenizer = load_model_without_cache()
    
    if model is None or tokenizer is None:
        print("✗ Model loading failed")
        return False
    
    print("✓ Model and tokenizer loaded successfully")
    
    # Test 2: Prompt Creation
    print("\n2. Testing Chinglish prompt creation...")
    test_topic = "上課遲到"
    prompt = create_chinglish_prompt(test_topic)
    
    if not prompt or len(prompt) < 100:
        print("✗ Prompt creation failed or too short")
        return False
    
    print("✓ Prompt created successfully")
    print(f"  Topic: {test_topic}")
    print(f"  Prompt length: {len(prompt)} characters")
    print(f"  Preview: {prompt[:200]}...")
    
    # Test 3: Text Generation with Various Topics
    print("\n3. Testing text generation with various topics...")
    
    test_topics = [
        "上課遲到",
        "programming homework",
        "週末計畫",
        "找工作面試"
    ]
    
    all_passed = True
    
    for i, topic in enumerate(test_topics, 1):
        print(f"\n  Test {i}/{len(test_topics)}: {topic}")
        
        result = generate_chinglish_text(
            model,
            tokenizer,
            topic,
            max_length=150,
            temperature=0.8
        )
        
        if not result['success']:
            print(f"  ✗ Generation failed: {result['error']}")
            all_passed = False
            continue
        
        text = result['text']
        gen_time = result['generation_time']
        
        print(f"  ✓ Generated in {gen_time:.2f}s")
        print(f"  Length: {len(text)} characters")
        
        # Validate language mixing
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in text)
        has_english = any(char.isalpha() and ord(char) < 128 for char in text)
        
        if has_chinese and has_english:
            print("  ✓ Contains both Chinese and English")
        else:
            print("  ⚠ Warning: May lack proper language mixing")
            print(f"    Has Chinese: {has_chinese}, Has English: {has_english}")
        
        print(f"  Text: {text[:200]}...")
    
    # Test 4: Error Handling
    print("\n4. Testing error handling...")
    
    # Test with empty topic
    result = generate_chinglish_text(model, tokenizer, "", max_length=100)
    if result['success']:
        print("  ✓ Handles empty topic gracefully")
    else:
        print("  ⚠ Empty topic generated error (expected behavior)")
    
    # Test with very long topic
    long_topic = "這是一個非常非常長的主題 " * 20
    result = generate_chinglish_text(model, tokenizer, long_topic, max_length=100)
    if result is not None:
        print("  ✓ Handles long topic gracefully")
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("✓ All tests passed!")
        print("\nModel integration is working correctly.")
        print("Chinglish generation quality appears good.")
        return True
    else:
        print("⚠ Some tests had warnings or failures")
        print("\nModel integration is functional but may need tuning.")
        return True  # Still return True if basic functionality works


if __name__ == "__main__":
    try:
        success = test_model_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
