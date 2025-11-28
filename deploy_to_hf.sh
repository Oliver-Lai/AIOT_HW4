#!/bin/bash

# Hugging Face Spaces 部署腳本

echo "=========================================="
echo "晶晶體生成器 - Hugging Face Spaces 部署"
echo "=========================================="
echo ""

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 檢查 huggingface_hub
if ! pip show huggingface_hub > /dev/null 2>&1; then
    echo -e "${YELLOW}正在安裝 huggingface_hub...${NC}"
    pip install huggingface_hub
fi

# 檢查登入狀態
echo "檢查 Hugging Face 登入狀態..."
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo -e "${YELLOW}需要登入 Hugging Face${NC}"
    echo ""
    echo "請按照以下步驟操作："
    echo "1. 訪問 https://huggingface.co/settings/tokens"
    echo "2. 創建一個新的 Access Token (write 權限)"
    echo "3. 複製 token"
    echo ""
    echo "現在執行登入..."
    huggingface-cli login
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}登入失敗！${NC}"
        exit 1
    fi
fi

# 獲取用戶名
USERNAME=$(huggingface-cli whoami 2>/dev/null | grep "username:" | awk '{print $2}')
echo -e "${GREEN}已登入為: $USERNAME${NC}"
echo ""

# Space 配置
read -p "請輸入 Space 名稱 (例如: chinglish-generator): " SPACE_NAME
SPACE_NAME=${SPACE_NAME:-chinglish-generator}

SPACE_ID="$USERNAME/$SPACE_NAME"
SPACE_URL="https://huggingface.co/spaces/$SPACE_ID"

echo ""
echo "將創建 Space: $SPACE_ID"
echo "URL: $SPACE_URL"
echo ""

read -p "繼續? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 創建 Space
echo ""
echo "正在創建 Hugging Face Space..."

python3 << EOF
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()

try:
    # 創建 Space
    create_repo(
        repo_id="$SPACE_ID",
        repo_type="space",
        space_sdk="streamlit",
        private=False,
        exist_ok=True
    )
    print("✓ Space 創建成功或已存在")
except Exception as e:
    print(f"創建 Space 時出錯: {e}")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}創建 Space 失敗！${NC}"
    exit 1
fi

# 準備檔案
echo ""
echo "準備部署檔案..."

# 創建臨時目錄
TEMP_DIR=$(mktemp -d)
echo "臨時目錄: $TEMP_DIR"

# 複製必要檔案
echo "複製檔案..."
cp app.py "$TEMP_DIR/"
cp model_utils.py "$TEMP_DIR/"
cp tts_utils.py "$TEMP_DIR/"
cp requirements.txt "$TEMP_DIR/"

# 使用 HF 專用 README
if [ -f "README_HF.md" ]; then
    cp README_HF.md "$TEMP_DIR/README.md"
else
    cp README.md "$TEMP_DIR/"
fi

# 創建 .gitignore (避免上傳大文件)
cat > "$TEMP_DIR/.gitignore" << 'GITIGNORE'
__pycache__/
*.pyc
models/
temp_audio/
*.mp3
.streamlit/secrets.toml
GITIGNORE

# 上傳到 Space
echo ""
echo "上傳檔案到 Hugging Face Space..."

cd "$TEMP_DIR"

python3 << EOF
from huggingface_hub import HfApi
import os

api = HfApi()

try:
    # 上傳整個目錄
    api.upload_folder(
        folder_path=".",
        repo_id="$SPACE_ID",
        repo_type="space",
        commit_message="Deploy Chinglish Generator to Hugging Face Spaces"
    )
    print("✓ 檔案上傳成功")
except Exception as e:
    print(f"上傳失敗: {e}")
    exit(1)
EOF

UPLOAD_STATUS=$?

# 清理
cd - > /dev/null
rm -rf "$TEMP_DIR"

if [ $UPLOAD_STATUS -ne 0 ]; then
    echo -e "${RED}上傳失敗！${NC}"
    exit 1
fi

# 完成
echo ""
echo "=========================================="
echo -e "${GREEN}部署成功！${NC}"
echo "=========================================="
echo ""
echo "你的 Space 正在構建中..."
echo ""
echo "Space URL: $SPACE_URL"
echo ""
echo "首次啟動需要 5-10 分鐘來："
echo "  1. 安裝依賴"
echo "  2. 下載模型（約 1GB）"
echo "  3. 啟動應用"
echo ""
echo "請訪問上述 URL 查看進度。"
echo ""
echo "構建日誌: $SPACE_URL/logs"
echo ""
echo "=========================================="
