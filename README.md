# 晶晶體生成器 Chinglish Text Generator

一個基於本地輕量級語言模型的 Web 應用，能夠生成「晶晶體」風格的文字（中英文混合表達）並轉換成語音。

## ✨ 功能特色

- 🤖 **智能文字生成**: 使用 Qwen2-0.5B-Instruct 模型生成自然的晶晶體文字
- 🎤 **語音合成**: 支援混合中英文的文字轉語音 (TTS)
- 🌐 **Web 介面**: 簡潔直觀的 Streamlit 界面
- 💾 **本地運行**: 所有模型預先下載，無需線上 API
- ⚙️ **可調參數**: 自定義生成長度、創意程度、語速等

## 🚀 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 下載模型

```bash
python scripts/download_models.py
```

模型會下載到 `models/qwen2-0.5b-instruct/` 目錄（約 3GB）。

### 3. 啟動應用

**方法 A: 使用啟動腳本（推薦）**
```bash
./run.sh
```

**方法 B: 直接使用 Streamlit**
```bash
streamlit run app.py
```

應用會在 `http://localhost:8501` 啟動。

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