# 🚀 Hugging Face Spaces 一鍵部署指南

## 📋 完整部署步驟

### 第一步：獲取 Hugging Face Token

1. **訪問 Token 頁面**
   ```
   https://huggingface.co/settings/tokens
   ```

2. **創建新 Token**
   - 點擊 "New token"
   - Name: `chinglish-generator-deploy`
   - Type: **Write** (重要！)
   - 點擊 "Generate"
   - **複製 token**（只會顯示一次）

### 第二步：登入 Hugging Face CLI

在終端執行：

```bash
huggingface-cli login
```

貼上你的 token，按 Enter。

**或者使用 Python：**

```bash
python -c "from huggingface_hub import login; login()"
```

### 第三步：執行部署腳本

```bash
python scripts/deploy_to_huggingface.py
```

按提示操作：
1. 確認已登入
2. 輸入 Space 名稱（預設：chinglish-generator）
3. 確認部署（輸入 y）
4. 等待上傳完成

### 第四步：等待構建

訪問你的 Space URL（腳本會顯示）：
```
https://huggingface.co/spaces/YOUR_USERNAME/chinglish-generator
```

首次構建需要 **5-10 分鐘**：
- ✅ 安裝依賴套件（1-2 分鐘）
- ✅ 下載模型（3-5 分鐘）
- ✅ 啟動應用（1-2 分鐘）

可以點擊 "Logs" 查看構建進度。

---

## 🎯 快速命令總結

```bash
# 1. 登入 Hugging Face
huggingface-cli login

# 2. 部署
python scripts/deploy_to_huggingface.py

# 3. 訪問你的 Space
# URL 會在部署完成後顯示
```

---

## 🔧 手動部署（備用方案）

如果自動腳本有問題，可以手動部署：

### 方法 A：通過 Web 界面

1. **創建 Space**
   - 訪問 https://huggingface.co/new-space
   - Space name: `chinglish-generator`
   - License: Apache 2.0
   - SDK: **Streamlit**
   - Hardware: CPU basic (free)
   - 點擊 "Create Space"

2. **上傳檔案**
   
   在 Space 的 "Files" 標籤中，上傳以下檔案：
   - `app.py`
   - `model_utils.py`
   - `tts_utils.py`
   - `requirements.txt`
   - `README.md`（使用 README_HF.md 的內容）

3. **等待構建**
   
   Space 會自動開始構建。

### 方法 B：通過 Git 克隆

```bash
# 1. 克隆 Space 倉庫
git clone https://huggingface.co/spaces/YOUR_USERNAME/chinglish-generator
cd chinglish-generator

# 2. 複製檔案
cp /path/to/AIOT_HW4/app.py .
cp /path/to/AIOT_HW4/model_utils.py .
cp /path/to/AIOT_HW4/tts_utils.py .
cp /path/to/AIOT_HW4/requirements.txt .
cp /path/to/AIOT_HW4/README_HF.md README.md

# 3. 提交並推送
git add .
git commit -m "Initial deployment"
git push
```

---

## 📊 部署後檢查

### 查看 Space 狀態

訪問你的 Space，應該看到：
- ✅ "Building" → 構建中
- ✅ "Running" → 運行中
- ❌ "Error" → 有錯誤（查看 Logs）

### 測試功能

1. 輸入主題（例如：週末計畫）
2. 點擊「生成晶晶體」
3. 查看生成的文字
4. 測試語音播放

### 查看日誌

點擊 Space 頁面的 "Logs" 標籤，可以看到：
- 模型下載進度
- 應用啟動日誌
- 錯誤訊息（如有）

---

## 🐛 常見問題

### Q1: "Not logged in" 錯誤

**解決方案：**
```bash
huggingface-cli login
# 輸入你的 token
```

### Q2: Token 權限不足

**解決方案：**
- 確保 token 類型是 "Write"（不是 Read）
- 重新創建一個 Write token

### Q3: 上傳失敗

**解決方案：**
```bash
# 檢查網路連接
ping huggingface.co

# 重新執行部署腳本
python scripts/deploy_to_huggingface.py
```

### Q4: Space 構建失敗

**解決方案：**
1. 查看 Logs 找出錯誤
2. 常見原因：
   - requirements.txt 格式錯誤
   - 程式碼錯誤
   - 相依套件衝突
3. 修正後重新上傳

### Q5: 記憶體不足錯誤

**解決方案：**
- HF Spaces 免費版提供 2GB RAM，應該足夠
- 如仍不足，可以：
  - 在 Settings 中升級硬體
  - 或使用更小的模型

---

## 🎨 自訂 Space

### 修改外觀

編輯 README.md 前言：

```yaml
---
title: 你的標題
emoji: 🎨  # 改成其他 emoji
colorFrom: blue
colorTo: purple
---
```

### 添加 Secrets

在 Space Settings > Variables and secrets：

```
MODEL_PATH=Qwen/Qwen2-0.5B-Instruct
```

### 升級硬體

Settings > Hardware：
- CPU basic（免費）
- CPU upgrade（付費）
- GPU（付費，更快）

---

## 📱 分享你的 Space

Space URL 格式：
```
https://huggingface.co/spaces/YOUR_USERNAME/chinglish-generator
```

可以：
- 直接分享連結
- 嵌入網站：
  ```html
  <iframe src="https://huggingface.co/spaces/YOUR_USERNAME/chinglish-generator" 
          width="100%" height="800"></iframe>
  ```
- 添加到個人 Profile

---

## 🔄 更新 Space

### 方法 A：重新執行部署腳本

```bash
python scripts/deploy_to_huggingface.py
```

選擇相同的 Space 名稱，會覆蓋舊檔案。

### 方法 B：通過 Git

```bash
cd your-space-directory
# 修改檔案
git add .
git commit -m "Update app"
git push
```

### 方法 C：通過 Web 界面

直接在 Space 的 Files 標籤中編輯或上傳檔案。

---

## 📈 監控使用情況

### 查看統計

Space Settings > Analytics：
- 訪問次數
- 使用時間
- 資源使用

### Space 睡眠機制

- 免費 Space 在 48 小時無活動後會睡眠
- 首次訪問喚醒約需 30 秒
- 升級為付費可保持常駐

---

## ✅ 部署完成檢查清單

- [ ] 已獲取 HF Token（Write 權限）
- [ ] 已登入 huggingface-cli
- [ ] 已執行部署腳本
- [ ] Space 創建成功
- [ ] 檔案上傳完成
- [ ] Space 構建成功（查看 Logs）
- [ ] 應用可正常訪問
- [ ] 測試文字生成功能
- [ ] 測試語音合成功能
- [ ] 已分享 Space URL

---

## 🎉 完成！

你的晶晶體生成器現在運行在 Hugging Face Spaces 上！

**下一步：**
- 🌟 給 Space 添加 Star
- 📢 分享給朋友
- 🔧 根據需求自訂功能
- 📊 查看使用統計

**需要幫助？**
- [HF Spaces 文檔](https://huggingface.co/docs/hub/spaces)
- [Streamlit 文檔](https://docs.streamlit.io/)
- [GitHub Issues](https://github.com/Oliver-Lai/AIOT_HW4/issues)

---

**祝部署順利！** 🚀
