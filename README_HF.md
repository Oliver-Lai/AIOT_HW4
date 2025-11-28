---
title: Chinglish Generator 晶晶體生成器
emoji: 🗣️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# 🗣️ 晶晶體生成器 Chinglish Text Generator

一個基於 AI 的應用，生成「晶晶體」風格的文字（中英文混合表達）並轉換成語音。

## 功能特色

- 🤖 **智能文字生成**: 使用 Qwen/Qwen2-0.5B-Instruct 輕量級模型
- 🎤 **語音合成**: 混合中英文 TTS
- 🌐 **Web 介面**: Streamlit 界面
- ⚙️ **可調參數**: 自定義生成設定

## 使用方法

1. 輸入主題（如：週末計畫、咖啡廳閒聊）
2. 調整參數（可選）
3. 點擊「生成晶晶體」
4. 查看文字並播放語音

## 什麼是晶晶體？

「晶晶體」是留學生或海外工作者常見的語言現象：
- 中英文自然混合
- 英文比例約 60-70%
- 英文用於名詞、動詞、形容詞
- 中文用於連接詞和語氣詞

**範例**：
> "我今天 really want to go shopping，但是 budget 有點 tight，所以可能只能 window shopping"

## 技術架構

- **模型**: Qwen/Qwen2-0.5B-Instruct (500M 參數，輕量高效)
- **TTS**: Google Text-to-Speech (gTTS)
- **框架**: Streamlit
- **深度學習**: PyTorch + Transformers

## 本地運行

```bash
# 安裝依賴
pip install -r requirements.txt

# 下載模型（可選，會自動下載）
python scripts/download_models.py

# 啟動應用
streamlit run app.py
```

## 源代碼

完整源代碼和文檔請見 [GitHub 倉庫](https://github.com/Oliver-Lai/AIOT_HW4)

## 授權

Apache 2.0 License
