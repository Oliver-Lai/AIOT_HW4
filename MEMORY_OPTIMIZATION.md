# Streamlit Cloud Memory Optimization Guide

## 問題：Resource Limits 錯誤

Streamlit Cloud 免費版限制：
- RAM: ~1GB
- CPU: 共享
- 磁碟空間: 有限

Qwen2-0.5B 模型載入後約需 1-1.5GB RAM，接近或超過限制。

## 已實施的優化 ✅

### 1. 模型載入優化
```python
- device_map="cpu"  # 強制使用 CPU
- low_cpu_mem_usage=True  # 啟用低記憶體模式
- use_cache=False  # 載入時禁用 KV cache
- model.eval()  # 設為評估模式
```

### 2. 生成過程優化
```python
- torch.no_grad()  # 禁用梯度計算
- max_new_tokens=150  # 限制生成長度
- num_beams=1  # 使用貪婪解碼
- truncation=True  # 截斷過長輸入
```

## 替代方案

### 方案 A：升級 Streamlit Cloud 方案（推薦）
- Community Cloud（付費）提供更多資源
- 價格：約 $20/月
- 可穩定運行此應用

### 方案 B：使用更小的模型
修改 `.streamlit/secrets.toml`：
```toml
MODEL_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 更小的模型
```

或使用純 API 方案（無需載入模型）：
```toml
USE_API = true
API_PROVIDER = "openai"  # 或其他 API
```

### 方案 C：部署到其他平台

#### 1. Hugging Face Spaces（推薦）
- 免費方案提供更多資源
- 支援 Gradio 和 Streamlit
- 步驟：
  ```bash
  # 1. 在 HF 創建 Space
  # 2. 上傳代碼
  # 3. 添加 requirements.txt
  # 4. 自動部署
  ```

#### 2. Railway.app
- 免費 $5 credit/月
- 更寬鬆的資源限制
- 支援 Docker 部署

#### 3. Render.com
- 免費方案：512MB RAM
- 更適合小型應用

#### 4. Google Cloud Run
- Pay-as-you-go
- 適合偶爾使用

### 方案 D：實施動態模型載入
只在需要時載入模型，使用後釋放：

```python
def generate_with_cleanup(topic):
    model, tokenizer = load_model()
    result = generate_text(model, tokenizer, topic)
    
    # 清理記憶體
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return result
```

缺點：每次生成都要重新載入模型（慢）

## 監控記憶體使用

添加到 app.py：
```python
import psutil
import gc

def show_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    st.sidebar.metric(
        "記憶體使用", 
        f"{mem_info.rss / 1024 / 1024:.1f} MB"
    )
```

## 建議的行動方案

### 短期方案（立即）
1. ✅ 已應用記憶體優化（當前修改）
2. 重新部署到 Streamlit Cloud
3. 測試是否能正常運行

### 中期方案（如果仍超限）
1. 使用 Hugging Face Spaces（免費且資源更多）
2. 或升級 Streamlit Cloud 方案

### 長期方案（最佳體驗）
1. 付費雲端方案
2. 自建伺服器
3. 使用 API 而非本地模型

## 測試當前優化

在本地測試記憶體使用：
```bash
python -c "
import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2-0.5B-Instruct',
    torch_dtype=torch.float32,
    device_map='cpu',
    low_cpu_mem_usage=True
)
import psutil
print(f'Memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## 更新部署

```bash
git add .
git commit -m "Add memory optimizations for Streamlit Cloud"
git push
```

Streamlit Cloud 會自動重新部署。

---

**如果優化後仍然超限，建議改用 Hugging Face Spaces 或升級方案。**
