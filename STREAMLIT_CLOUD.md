# Streamlit Cloud 部署指南

## 🚀 快速部署步驟

### 1. 推送代碼到 GitHub

```bash
# 確保已初始化 Git 倉庫
git init

# 添加所有文件（models/ 會被 .gitignore 排除）
git add .

# 提交
git commit -m "Add Chinglish Generator app"

# 推送到 GitHub
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

### 2. 部署到 Streamlit Cloud

1. 訪問 https://share.streamlit.io/
2. 使用 GitHub 帳號登入
3. 點擊 "New app"
4. 選擇：
   - **Repository**: 你的 GitHub 倉庫
   - **Branch**: main
   - **Main file path**: app.py
5. 點擊 "Deploy"

### 3. 等待部署完成

- 首次部署需要 5-10 分鐘
- 應用會自動從 Hugging Face Hub 下載模型（約 1GB）
- 完成後會獲得一個公開 URL

## ⚙️ 環境配置（可選）

如果需要自定義設定，在 Streamlit Cloud 的 "Advanced settings" 中添加 Secrets：

```toml
# 使用特定模型
MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"

# 如果使用私有模型（需要 Hugging Face token）
# HUGGING_FACE_TOKEN = "hf_xxxxxxxxxxxxx"
```

## 📋 注意事項

### ✅ 優點
- **無需上傳模型**: 自動從 Hugging Face Hub 下載
- **快速部署**: 幾分鐘內完成
- **免費託管**: Streamlit Cloud 提供免費方案
- **自動快取**: 模型下載後會被快取

### ⚠️ 限制
- **首次啟動慢**: 需要下載 1GB 模型（3-5 分鐘）
- **記憶體限制**: Streamlit Cloud 免費版有 1GB RAM 限制
  - 使用 Qwen2-0.5B 是可行的
  - 如果遇到問題，考慮升級方案
- **需要網路**: TTS 功能需要連接 Google 服務

### 💡 優化建議

#### 1. 減少記憶體使用

在 `model_utils.py` 中已經添加了 `low_cpu_mem_usage=True`

#### 2. 使用更小的模型（如果需要）

可以在 Secrets 中設定：
```toml
MODEL_PATH = "Qwen/Qwen2-0.5B-Instruct"  # 最小模型
```

#### 3. 監控應用狀態

在 Streamlit Cloud Dashboard 中可以：
- 查看日誌
- 重啟應用
- 檢查資源使用情況

## 🔧 故障排除

### 問題 1: 應用啟動失敗（記憶體不足）

**解決方案**:
1. 升級到 Streamlit Cloud 付費方案（更多記憶體）
2. 或使用其他部署平台（Hugging Face Spaces, AWS, Google Cloud）

### 問題 2: 模型下載超時

**解決方案**:
1. 重新部署應用
2. 檢查 Hugging Face Hub 是否可訪問
3. 確認模型名稱正確

### 問題 3: TTS 功能失敗

**解決方案**:
1. gTTS 需要網路連接到 Google 服務
2. 某些地區可能無法訪問
3. 考慮使用其他 TTS 方案

## 📊 部署檢查清單

- [ ] 代碼已推送到 GitHub
- [ ] `.gitignore` 正確配置（排除 models/）
- [ ] requirements.txt 包含所有依賴
- [ ] app.py 在根目錄
- [ ] 已在 Streamlit Cloud 創建應用
- [ ] （可選）已配置 Secrets
- [ ] 首次啟動等待模型下載完成
- [ ] 測試文字生成功能
- [ ] 測試語音合成功能

## 🌐 部署後

### 獲取應用 URL

部署完成後，你會得到一個 URL：
```
https://your-app-name.streamlit.app
```

### 分享應用

- 可以直接分享 URL
- 設定自定義域名（付費功能）
- 添加密碼保護（在 Streamlit Cloud 設定中）

## 📈 監控與維護

### 查看日誌
```
Streamlit Cloud Dashboard > Your App > Logs
```

### 重啟應用
```
Streamlit Cloud Dashboard > Your App > ⋮ > Reboot app
```

### 更新應用

只需推送新代碼到 GitHub，Streamlit Cloud 會自動重新部署：
```bash
git add .
git commit -m "Update app"
git push
```

## 🔗 相關資源

- [Streamlit Cloud 文檔](https://docs.streamlit.io/streamlit-community-cloud)
- [Hugging Face Hub](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)
- [gTTS 文檔](https://gtts.readthedocs.io/)

---

**祝部署順利！** 🎉

如有問題，請參考 [DEPLOYMENT.md](DEPLOYMENT.md) 了解更多部署選項。
