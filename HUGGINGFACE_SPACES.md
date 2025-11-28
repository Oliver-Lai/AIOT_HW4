# 🤗 Hugging Face Spaces 部署指南（推薦）

## 為什麼選擇 Hugging Face Spaces？

✅ **優勢：**
- **更多資源**：免費方案提供 2GB RAM + 2 vCPU
- **原生支援**：專為 ML 應用設計
- **易於部署**：與 HF Hub 無縫整合
- **社群活躍**：大量範例和支援
- **完全免費**：無需信用卡

❌ **Streamlit Cloud 限制：**
- 免費版僅 ~1GB RAM
- 此應用模型需 1-1.5GB
- 容易超過資源限制

---

## 🚀 部署步驟

### 方法 A：從 GitHub 自動同步（推薦）

#### 1. 準備 GitHub 倉庫
```bash
# 確保代碼已推送
git add .
git commit -m "Prepare for HF Spaces deployment"
git push
```

#### 2. 創建 Hugging Face Space

1. 訪問 https://huggingface.co/new-space
2. 填寫資訊：
   - **Space name**: chinglish-generator（或自訂）
   - **License**: Apache 2.0
   - **SDK**: Streamlit
   - **Hardware**: CPU basic (free)
3. 點擊 "Create Space"

#### 3. 連接 GitHub 倉庫

在 Space 的 Settings 中：
1. 找到 "Repository" 區塊
2. 點擊 "Link to GitHub"
3. 選擇你的 GitHub 倉庫
4. 啟用自動同步

#### 4. 配置檔案

確保根目錄有這些檔案：

**requirements.txt**（已有）
```txt
streamlit>=1.28.0
transformers>=4.35.0
torch>=2.0.0
gtts>=2.5.0
accelerate>=0.24.0
```

**創建 `packages.txt`**（系統依賴，如需要）
```txt
# 通常不需要，但如有問題可添加
```

**創建 `README.md`（Space 頁面說明）**
HF Spaces 會自動顯示 README.md 作為首頁。

#### 5. 部署完成

- HF Spaces 會自動：
  1. 讀取 requirements.txt
  2. 安裝依賴
  3. 從 HF Hub 下載模型
  4. 啟動 Streamlit
- 首次部署約 5-10 分鐘
- 完成後獲得 URL：`https://huggingface.co/spaces/YOUR_USERNAME/chinglish-generator`

---

### 方法 B：直接上傳到 HF Spaces

#### 1. 克隆 Space 倉庫
```bash
# 安裝 huggingface_hub
pip install huggingface_hub

# 登入
huggingface-cli login

# 克隆 Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/chinglish-generator
cd chinglish-generator
```

#### 2. 複製檔案
```bash
# 從你的專案複製必要檔案
cp /path/to/your/project/app.py .
cp /path/to/your/project/model_utils.py .
cp /path/to/your/project/tts_utils.py .
cp /path/to/your/project/requirements.txt .
```

#### 3. 創建 Space README
```markdown
---
title: Chinglish Generator
emoji: 🗣️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.28.0"
app_file: app.py
pinned: false
---

# 晶晶體生成器

[應用說明...]
```

#### 4. 推送到 HF Spaces
```bash
git add .
git commit -m "Initial deployment"
git push
```

---

## 🔧 配置選項

### Space 元數據（README.md 前言）

```yaml
---
title: Chinglish Generator  # 顯示名稱
emoji: 🗣️  # 顯示圖標
colorFrom: blue  # 漸變起始顏色
colorTo: purple  # 漸變結束顏色
sdk: streamlit  # 使用 Streamlit
sdk_version: "1.28.0"  # Streamlit 版本
app_file: app.py  # 主檔案
pinned: false  # 是否置頂
license: apache-2.0  # 授權
---
```

### Secrets 設定（如需要）

在 Space Settings > Variables and secrets 中添加：
```
MODEL_PATH=Qwen/Qwen2-0.5B-Instruct
```

### 硬體選擇

- **CPU basic**（免費）：2GB RAM + 2 vCPU
  - 適合此應用
- **CPU upgrade**（付費）：更多資源
- **GPU**（付費）：如需要更快推理

---

## 📊 部署後

### 監控應用

在 Space 頁面：
- **Logs**：查看運行日誌
- **Settings**：配置選項
- **Files**：查看檔案

### 更新應用

**從 GitHub 自動同步：**
```bash
git push  # 推送到 GitHub，HF 會自動更新
```

**直接更新 Space：**
```bash
cd your-space-directory
# 修改檔案
git add .
git commit -m "Update"
git push
```

### 分享應用

你的 Space URL：
```
https://huggingface.co/spaces/YOUR_USERNAME/chinglish-generator
```

可以：
- 直接分享連結
- 嵌入到網站：`<iframe src="https://..."></iframe>`
- 添加到 HF Profile

---

## 🐛 常見問題

### Q1: Space 啟動失敗
**A**: 查看 Logs，常見原因：
- requirements.txt 錯誤
- 代碼錯誤
- 模型下載失敗

### Q2: 記憶體不足
**A**: HF Spaces 免費版提供 2GB，通常足夠。如仍不足：
- 檢查代碼是否有記憶體洩漏
- 考慮升級到付費硬體
- 使用更小的模型

### Q3: 模型下載很慢
**A**: 首次啟動需下載 ~1GB 模型，約 3-5 分鐘。之後會快取。

### Q4: 如何重啟 Space
**A**: Settings > Factory reboot

### Q5: Space 進入睡眠
**A**: 免費 Space 閒置 48 小時後會睡眠。首次訪問會喚醒（約 30 秒）。

---

## 📝 完整部署檢查清單

- [ ] 代碼已推送到 GitHub
- [ ] 創建 HF Space
- [ ] 選擇 Streamlit SDK
- [ ] 連接 GitHub 倉庫（或直接上傳）
- [ ] 確認 requirements.txt 正確
- [ ] 添加 Space README（可選）
- [ ] 配置 Secrets（如需要）
- [ ] 等待首次部署完成（5-10 分鐘）
- [ ] 測試應用功能
- [ ] 分享 URL

---

## 🎉 完成

你的晶晶體生成器現在運行在 Hugging Face Spaces 上！

**優勢總結：**
- ✅ 無記憶體限制問題
- ✅ 更快的啟動速度
- ✅ 更好的社群支援
- ✅ 完全免費

**下一步：**
- 自訂 Space 外觀
- 添加更多功能
- 分享給朋友使用

---

## 📚 參考資源

- [HF Spaces 官方文檔](https://huggingface.co/docs/hub/spaces)
- [Streamlit on HF Spaces](https://huggingface.co/docs/hub/spaces-sdks-streamlit)
- [HF Spaces 範例](https://huggingface.co/spaces)
