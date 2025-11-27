# 晶晶體生成器 - 部署指南

## 📋 目錄

1. [本地部署](#本地部署)
2. [Streamlit Cloud 部署](#streamlit-cloud-部署)
3. [Docker 部署](#docker-部署)
4. [故障排除](#故障排除)

---

## 🏠 本地部署

### 前置需求

- Python 3.8 或更高版本
- pip 套件管理器
- 至少 4GB RAM
- 5GB 可用磁碟空間

### 安裝步驟

#### 1. 克隆專案

```bash
git clone <your-repo-url>
cd AIOT_HW4
```

#### 2. 建立虛擬環境（建議）

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. 安裝依賴

```bash
pip install -r requirements.txt
```

#### 4. 下載模型

```bash
python scripts/download_models.py
```

這會下載約 3GB 的模型檔案到 `models/` 目錄。

#### 5. 執行測試（可選）

```bash
# 測試模型載入
python scripts/test_model_loading.py

# 測試文字生成
python scripts/test_generation.py

# 測試 TTS 功能
python scripts/test_actual_tts.py

# 完整整合測試
python scripts/test_integration.py
```

#### 6. 啟動應用

**方法 A: 使用啟動腳本**
```bash
./run.sh
```

**方法 B: 直接使用 Streamlit**
```bash
streamlit run app.py
```

**方法 C: 自訂端口**
```bash
streamlit run app.py --server.port 8080
```

應用會自動在瀏覽器中打開，或訪問：
- 本地: http://localhost:8501
- 網路: http://<your-ip>:8501

---

## ☁️ Streamlit Cloud 部署

### 準備工作

由於模型檔案較大（3GB），不適合直接部署到 Streamlit Cloud。建議使用以下方案之一：

### 方案 A: 使用 Hugging Face Hub（推薦）

修改 `model_utils.py` 中的 `load_model()` 函數：

```python
@st.cache_resource
def load_model(model_path: str = "Qwen/Qwen2-0.5B-Instruct"):
    """Load model from Hugging Face Hub instead of local path."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        
        return model, tokenizer
    except Exception as e:
        st.error(f"模型載入失敗: {e}")
        return None, None
```

### 方案 B: 使用外部模型存儲

1. 將模型上傳到雲端存儲（如 AWS S3, Google Cloud Storage）
2. 在應用啟動時下載模型
3. 使用 Streamlit Secrets 管理存儲憑證

### 部署步驟

1. **推送代碼到 GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo>
   git push -u origin main
   ```

2. **連接到 Streamlit Cloud**
   - 訪問 https://share.streamlit.io/
   - 登入並連接 GitHub 帳戶
   - 選擇你的倉庫和分支
   - 選擇 `app.py` 作為主檔案

3. **配置 Secrets（如需要）**
   
   在 Streamlit Cloud 的設定中添加：
   ```toml
   [huggingface]
   token = "your_hf_token_here"
   ```

4. **部署**
   
   點擊 "Deploy" 按鈕，等待部署完成。

---

## 🐳 Docker 部署

### Dockerfile

創建 `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Download models (optional - can be mounted as volume)
RUN python scripts/download_models.py

# Create audio directory
RUN mkdir -p temp_audio

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

創建 `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chinglish-generator:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models  # Mount models directory
      - ./temp_audio:/app/temp_audio  # Mount audio directory
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

### 構建和運行

```bash
# 構建鏡像
docker build -t chinglish-generator .

# 運行容器
docker run -p 8501:8501 chinglish-generator

# 或使用 docker-compose
docker-compose up -d
```

### 使用預先下載的模型

為了減少鏡像大小，可以將模型作為卷掛載：

```bash
# 在主機上下載模型
python scripts/download_models.py

# 運行容器並掛載模型目錄
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/temp_audio:/app/temp_audio \
  chinglish-generator
```

---

## 🔧 故障排除

### 問題 1: 模型載入失敗

**錯誤訊息**: "模型載入失敗" 或 "找不到模型檔案"

**解決方案**:
```bash
# 檢查模型目錄
ls -la models/qwen2-0.5b-instruct/

# 重新下載模型
rm -rf models/qwen2-0.5b-instruct
python scripts/download_models.py
```

### 問題 2: 記憶體不足

**錯誤訊息**: "CUDA out of memory" 或 "RuntimeError: out of memory"

**解決方案**:
1. 減少 `max_length` 參數
2. 關閉其他應用程式
3. 使用更小的模型或 CPU 模式

```python
# 強制使用 CPU
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu"
)
```

### 問題 3: TTS 失敗

**錯誤訊息**: "網路連線失敗" 或 "gTTS 請求超時"

**解決方案**:
1. 檢查網路連線
2. 確認可以訪問 Google 服務
3. 嘗試使用 VPN（如果在受限網路環境）
4. 考慮使用離線 TTS 方案（如 pyttsx3）

### 問題 4: Streamlit 埠被占用

**錯誤訊息**: "Port 8501 is already in use"

**解決方案**:
```bash
# 找出占用埠的進程
lsof -i :8501

# 終止進程
kill -9 <PID>

# 或使用不同埠
streamlit run app.py --server.port 8502
```

### 問題 5: 依賴安裝失敗

**錯誤訊息**: pip 安裝錯誤

**解決方案**:
```bash
# 升級 pip
pip install --upgrade pip

# 逐個安裝有問題的套件
pip install streamlit
pip install transformers
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU版本

# 檢查 Python 版本
python --version  # 需要 3.8+
```

---

## 🌐 生產環境部署建議

### 1. 使用反向代理（Nginx）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 2. 啟用 HTTPS

```bash
# 使用 Let's Encrypt
sudo certbot --nginx -d your-domain.com
```

### 3. 設定 Systemd 服務

創建 `/etc/systemd/system/chinglish-generator.service`:

```ini
[Unit]
Description=Chinglish Generator
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/AIOT_HW4
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/streamlit run app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

啟用服務:
```bash
sudo systemctl daemon-reload
sudo systemctl enable chinglish-generator
sudo systemctl start chinglish-generator
```

### 4. 監控和日誌

```bash
# 查看日誌
sudo journalctl -u chinglish-generator -f

# 或使用 Streamlit 日誌
tail -f ~/.streamlit/logs/*.log
```

---

## 📊 性能優化

### 1. 模型量化

減少模型大小和推理時間：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,  # 8-bit 量化
    device_map="auto"
)
```

### 2. 批次處理

如果有多個請求，可以考慮批次處理。

### 3. 快取策略

已使用 `@st.cache_resource` 快取模型和 `@st.cache_data` 快取數據。

### 4. 音訊檔案清理

定期清理舊的音訊檔案：

```bash
# 添加 cron job
0 * * * * find /path/to/temp_audio -name "*.mp3" -mtime +1 -delete
```

---

## 🔒 安全性建議

1. **不要將模型檔案提交到 Git**
   ```bash
   # .gitignore
   models/
   temp_audio/
   *.mp3
   ```

2. **使用環境變數管理敏感資訊**
   ```python
   import os
   hf_token = os.getenv("HF_TOKEN")
   ```

3. **限制上傳大小和請求頻率**

4. **定期更新依賴套件**
   ```bash
   pip list --outdated
   pip install --upgrade <package>
   ```

---

## 📞 支援

如有問題，請：
1. 查看 [README.md](README.md)
2. 檢查 [Issues](https://github.com/your-repo/issues)
3. 提交新的 Issue

---

**最後更新**: 2024-11-27
