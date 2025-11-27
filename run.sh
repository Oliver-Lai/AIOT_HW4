#!/bin/bash

# Chinglish Generator - Launch Script
# This script starts the Streamlit application

echo "=========================================="
echo "晶晶體生成器 Chinglish Generator"
echo "=========================================="
echo ""

# Check if required files exist
if [ ! -f "app.py" ]; then
    echo "❌ 錯誤: app.py 不存在"
    exit 1
fi

if [ ! -f "model_utils.py" ]; then
    echo "❌ 錯誤: model_utils.py 不存在"
    exit 1
fi

if [ ! -f "tts_utils.py" ]; then
    echo "❌ 錯誤: tts_utils.py 不存在"
    exit 1
fi

# Check if models directory exists
if [ ! -d "models/qwen2-0.5b-instruct" ]; then
    echo "⚠️  警告: 模型目錄不存在"
    echo "請先執行: python scripts/download_models.py"
    echo ""
    read -p "是否現在下載模型？ (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/download_models.py
        if [ $? -ne 0 ]; then
            echo "❌ 模型下載失敗"
            exit 1
        fi
    else
        echo "跳過模型下載，應用可能無法正常運作"
    fi
fi

# Create temp_audio directory if it doesn't exist
mkdir -p temp_audio

echo ""
echo "✓ 所有檢查通過"
echo ""
echo "啟動 Streamlit 應用..."
echo "=========================================="
echo ""

# Run Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
