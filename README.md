# 晶晶體生成器 Chinglish Text Generator

## 📝 項目摘要

本專案是一個創新的 AI 應用，專門生成「晶晶體」風格的文字內容並轉換為語音。晶晶體是留學生或海外工作者常見的語言現象，特色是中英文自然混合使用，英文比例約 60-70%，主要用於名詞和動詞，中文則用於連接詞和語氣詞。

系統採用 Qwen2-0.5B-Instruct 輕量級語言模型，結合精心設計的 Prompt 工程，能夠根據用戶輸入的主題（如「週末計畫」、「咖啡廳閒聊」）生成自然流暢的晶晶體文字。生成的內容會透過 Google Text-to-Speech (gTTS) 引擎轉換為語音，讓使用者可以直接聆聽和下載。

整個應用基於 Streamlit 框架開發，提供簡潔直觀的 Web 介面，支援本地部署和雲端部署兩種模式。本地部署使用預先下載的模型檔案（約 3GB），而 Streamlit Cloud 部署則自動從 Hugging Face Hub 下載模型，無需手動處理大型檔案。系統包含完整的錯誤處理、超時保護、記憶體優化等機制，並通過 5 個測試腳本驗證，測試通過率 100%。

專案包含超過 1,500 行 Python 代碼和 1,200 行文檔，涵蓋開發、測試、部署的完整流程，適合作為 NLP 和 TTS 整合應用的學習範例。

## 🤖 Agent 開發過程

### 第一階段：需求分析與提案設計
**使用者需求**：建立一個可以調用本地輕量語言模型的網頁，使用者輸入主題後，語言模型生成晶晶體風格的文字，並轉換成語音。

**Agent 回應**：
- 創建完整的 OpenSpec 提案文檔（proposal.md, design.md, tasks.md）
- 定義 3 個核心功能規格：文字生成、語音合成、Web 介面
- 規劃 43 個詳細實現任務，分為 8 個階段
- 技術選型：從 Ollama + FastAPI 修改為 Streamlit + Hugging Face Transformers

### 第二階段：環境設置與模型整合（Step 1-2）
**使用者需求**：實現 Step 1 和 Step 2，已有部分初始代碼。

**Agent 執行**：
1. 創建項目目錄結構（models/, temp_audio/, scripts/）
2. 撰寫 requirements.txt 包含所有依賴
3. 實現 scripts/download_models.py 下載 Qwen2-0.5B-Instruct
4. 成功下載模型（11 個檔案，約 3GB）
5. 實現 model_utils.py：
   - `load_model()` 函數使用 @st.cache_resource 快取
   - `create_chinglish_prompt()` 設計系統提示詞
   - `generate_chinglish_text()` 文字生成函數
6. 添加錯誤處理、超時機制（60秒）、重試邏輯
7. 創建測試腳本驗證功能

### 第三階段：Prompt 優化與 TTS 整合（Step 3）
**使用者需求**：改進 prompt 提高英文含量，完成 Step 3 語音合成。

**Agent 執行**：
1. 優化 `create_chinglish_prompt()` 函數：
   - 提高英文比例從 30-40% 到 60-70%
   - 添加 4 個高品質範例句子
   - 明確規則：英文用於名詞/動詞/形容詞，中文僅用於連接詞
2. 創建 tts_utils.py 模組（285 行）：
   - `synthesize_speech()` 語音合成函數
   - `detect_language_segments()` 語言片段偵測
   - `ensure_audio_directory()` 音訊目錄管理
   - `generate_audio_filename()` UUID/時間戳命名
   - `cleanup_old_audio_files()` 自動清理功能
3. 實現實際 TTS 測試，5 種文字類型全部通過
4. 平均合成時間：0.18 秒，檔案大小：18-58 KB

### 第四階段：Streamlit Web 應用開發（Step 4）
**Agent 執行**：
1. 創建 app.py 主應用程式（287 行）
2. 實現完整功能：
   - 精美的使用者介面（標題、說明、示範）
   - 側邊欄參數設定（長度、溫度、多樣性、慢速語音）
   - 主題輸入框與快速選擇按鈕
   - 文字生成與顯示
   - 即時音訊播放器
   - 音訊下載功能
   - 統計資訊顯示（字元數、生成時間、英文比例）
   - 音訊檔案管理（顯示數量、大小、清理）
3. 配置 .streamlit/config.toml 客製化主題
4. 創建 run.sh 啟動腳本
5. 成功啟動應用於 localhost:8501

### 第五階段：測試與驗證（Step 5）
**Agent 執行**：
1. 創建 5 個測試腳本：
   - test_model_loading.py：模型載入測試
   - test_generation.py：文字生成測試（4 個主題）
   - test_tts.py：TTS 功能測試（6 個測試項目）
   - test_actual_tts.py：實際語音合成測試（5 個樣本）
   - test_integration.py：完整整合測試
2. 所有測試 100% 通過
3. 驗證結果：
   - 模型載入成功
   - 文字生成時間：5-19 秒
   - 語音合成時間：0.15-0.22 秒
   - 英文比例達標：60-75%

### 第六階段：文檔撰寫
**Agent 執行**：
1. README.md：完整使用說明（150+ 行）
2. QUICKSTART.md：5 分鐘快速上手指南
3. DEPLOYMENT.md：詳細部署指南（400+ 行）
   - 本地部署步驟
   - Docker 部署配置
   - 生產環境建議
   - 故障排除指南
4. PROJECT_SUMMARY.md：項目總結報告
   - 技術架構說明
   - 代碼統計分析
   - 測試結果總結
   - 性能指標記錄

### 第七階段：Streamlit Cloud 部署支援
**使用者問題**：部署到 Streamlit 時出現「模型目錄不存在」錯誤。

**Agent 分析**：
- 問題原因：模型檔案 3GB 無法上傳到 GitHub
- Streamlit Cloud 沒有本地模型檔案
- 原代碼只支援本地路徑載入

**Agent 解決方案**：
1. 修改 model_utils.py 的 `load_model()` 函數：
   - 添加智能路徑檢測（本地 vs 雲端）
   - 支援從 Hugging Face Hub 自動下載
   - 添加環境變數配置（MODEL_PATH）
   - 記憶體優化（low_cpu_mem_usage=True）
   - 改進錯誤提示和進度顯示
2. 創建雲端部署文檔：
   - STREAMLIT_CLOUD.md：詳細部署步驟
   - DEPLOYMENT_FIX.md：問題解決說明
   - .streamlit/secrets.toml.example：配置範例
3. 創建 test_deployment_mode.py 驗證部署模式
4. 更新 README.md 添加雲端部署說明
5. 確認 .gitignore 正確配置（排除大型檔案）

**最終結果**：
- 支援本地部署（使用本地模型，2-5 秒啟動）
- 支援雲端部署（自動下載，首次 3-5 分鐘，後續 30 秒）
- 所有 43 個任務 100% 完成

### 開發統計
- **總代碼行數**：1,887 行 Python
- **總文檔行數**：1,213 行 Markdown
- **核心模組**：3 個（app.py, model_utils.py, tts_utils.py）
- **測試腳本**：6 個（包含部署測試）
- **文檔檔案**：7 個（README, QUICKSTART, DEPLOYMENT 等）
- **開發時間**：約 6-8 小時（包含測試與文檔）
- **測試通過率**：100%

## ✨ 功能特色

- 🤖 **智能文字生成**: 使用 Qwen2-0.5B-Instruct 模型生成自然的晶晶體文字
- 🎤 **語音合成**: 支援混合中英文的文字轉語音 (TTS)
- 🌐 **Web 介面**: 簡潔直觀的 Streamlit 界面
- 💾 **本地運行**: 所有模型預先下載，無需線上 API
- ⚙️ **可調參數**: 自定義生成長度、創意程度、語速等

## 🚀 快速開始

### 本地部署

#### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

#### 2. 下載模型（本地部署需要）

```bash
python scripts/download_models.py
```

模型會下載到 `models/qwen2-0.5b-instruct/` 目錄（約 3GB）。

#### 3. 啟動應用

**方法 A: 使用啟動腳本（推薦）**
```bash
./run.sh
```

**方法 B: 直接使用 Streamlit**
```bash
streamlit run app.py
```

應用會在 `http://localhost:8501` 啟動。

### ☁️ Streamlit Cloud 部署

**無需下載模型！** 應用會自動從 Hugging Face Hub 下載。

```bash
# 1. 推送到 GitHub
git add .
git commit -m "Add Chinglish Generator"
git push

# 2. 訪問 https://share.streamlit.io/
# 3. 選擇你的倉庫和 app.py
# 4. 點擊 Deploy
```

詳細步驟請見 [STREAMLIT_CLOUD.md](STREAMLIT_CLOUD.md)

## 📖 使用說明

1. **輸入主題**: 在文字框中輸入任何主題（如：週末計畫、咖啡廳閒聊）
2. **調整設定**: 在側邊欄調整文字生成和語音參數
3. **生成**: 點擊「✨ 生成晶晶體」按鈕
4. **享受結果**: 查看生成的文字並播放語音
5. **下載**: 可下載生成的音訊檔案

## 🛠️ 技術架構

### 核心組件

- **語言模型**: Qwen2-0.5B-Instruct (Hugging Face Transformers)
- **TTS 引擎**: Google Text-to-Speech (gTTS)
- **Web 框架**: Streamlit
- **深度學習**: PyTorch

### 項目結構

```
AIOT_HW4/
├── app.py                  # Streamlit 主應用
├── model_utils.py          # 模型加載與文字生成
├── tts_utils.py            # 語音合成功能
├── requirements.txt        # Python 依賴
├── run.sh                  # 啟動腳本
├── models/                 # 模型檔案目錄
│   └── qwen2-0.5b-instruct/
├── temp_audio/             # 暫存音訊檔案
├── scripts/                # 工具腳本
│   ├── download_models.py
│   ├── test_model_loading.py
│   ├── test_generation.py
│   ├── test_tts.py
│   └── test_actual_tts.py
├── .streamlit/             # Streamlit 配置
│   └── config.toml
└── openspec/               # 項目規格文件
    └── changes/add-chinglish-generator/
```

## 🧪 測試

### 測試模型載入
```bash
python scripts/test_model_loading.py
```

### 測試文字生成
```bash
python scripts/test_generation.py
```

### 測試 TTS 功能
```bash
python scripts/test_tts.py
```

### 測試實際語音合成
```bash
python scripts/test_actual_tts.py
```

## ⚙️ 配置選項

### 文字生成參數

- **最大長度** (50-300): 控制生成文字的長度
- **創意程度** (0.5-1.5): 控制文字的創造性
- **多樣性** (0.5-1.0): 控制輸出的多樣性

### 語音合成參數

- **慢速語音**: 生成較慢、更清晰的語音
- **語言**: 自動偵測混合中英文

## 📝 什麼是晶晶體？

「晶晶體」是一種語言現象，常見於留學生或海外工作者。特點：

- 💬 中英文混合使用
- 🔤 英文名詞、動詞為主
- 🔗 中文連接詞、語氣詞
- 📊 英文比例約 60-70%

**範例:**
> "我今天 really want to go shopping，但是 budget 有點 tight，所以可能只能 window shopping"

## 🔧 系統需求

- Python 3.8+
- 至少 4GB RAM
- 5GB 磁碟空間（模型 + 依賴）
- 網路連線（僅 TTS 需要）

## 📦 依賴套件

主要依賴：
- `streamlit >= 1.28.0`
- `transformers >= 4.35.0`
- `torch >= 2.0.0`
- `gtts >= 2.5.0`
- `accelerate >= 0.24.0`

完整列表見 `requirements.txt`

## ⚠️ 注意事項

1. **模型下載**: 首次使用需下載約 3GB 的模型檔案
2. **網路需求**: TTS 功能需要網路連線（使用 Google API）
3. **音訊管理**: 定期清理 `temp_audio/` 目錄以節省空間
4. **記憶體**: 建議至少 4GB RAM 以順暢運行

## 🐛 故障排除

### 模型載入失敗
```bash
# 重新下載模型
rm -rf models/qwen2-0.5b-instruct
python scripts/download_models.py
```

### TTS 失敗
- 檢查網路連線
- 確認 gTTS 套件已正確安裝
- 查看錯誤訊息，可能是 Google API 限制

### 記憶體不足
- 使用更小的 max_length 參數
- 關閉其他應用程式
- 考慮使用更大記憶體的機器

## 📄 授權

MIT License

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📧 聯絡

如有問題或建議，請開 Issue 討論。

---

**Made with ❤️ using Streamlit**