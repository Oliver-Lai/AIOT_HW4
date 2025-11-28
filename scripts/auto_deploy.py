"""
自動部署到 Hugging Face Spaces (無需互動)
"""

import os
import sys
from pathlib import Path
import shutil

try:
    from huggingface_hub import HfApi, create_repo, whoami
except ImportError:
    print("❌ 需要安裝 huggingface_hub")
    print("執行: pip install huggingface_hub")
    sys.exit(1)


def check_login():
    """檢查是否已登入 Hugging Face"""
    try:
        user_info = whoami()
        username = user_info.get('name', 'Unknown')
        print(f"✓ 已登入為: {username}")
        return username
    except Exception:
        print("❌ 未登入 Hugging Face")
        print("\n請執行: huggingface-cli login")
        sys.exit(1)


def create_or_update_space(space_id):
    """創建或更新 Hugging Face Space"""
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True  # 如果已存在則更新
        )
        print(f"✓ Space 準備就緒: {space_id}")
        return True
    except Exception as e:
        print(f"❌ Space 操作失敗: {e}")
        return False


def prepare_files(temp_dir):
    """準備要上傳的檔案"""
    project_root = Path(__file__).parent.parent
    
    # 要複製的檔案
    files_to_copy = [
        'app.py',
        'model_utils.py', 
        'tts_utils.py',
        'requirements.txt',
        'Dockerfile',
        'README_HF.md'
    ]
    
    print("\n準備檔案:")
    for file in files_to_copy:
        src = project_root / file
        if src.exists():
            # README_HF.md 複製為 README.md
            if file == 'README_HF.md':
                dst = temp_dir / 'README.md'
            else:
                dst = temp_dir / file
            shutil.copy2(src, dst)
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️  {file} 不存在")
    
    return True


def upload_to_space(space_id, temp_dir):
    """上傳檔案到 Space"""
    print(f"\n上傳到 Space: {space_id}")
    
    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path=str(temp_dir),
            repo_id=space_id,
            repo_type="space",
            commit_message="Update Chinglish Generator - Fix torch import and UI bugs"
        )
        print("✓ 上傳成功")
        return True
    except Exception as e:
        print(f"❌ 上傳失敗: {e}")
        return False


def main():
    print("=" * 60)
    print("晶晶體生成器 - 自動部署到 Hugging Face Spaces")
    print("=" * 60)
    print()
    
    # 1. 檢查登入
    print("步驟 1: 檢查登入狀態")
    username = check_login()
    print()
    
    # 2. 設定 Space (使用環境變數或預設值)
    space_name = os.getenv('HF_SPACE_NAME', 'chinglish-generator')
    space_id = f"{username}/{space_name}"
    space_url = f"https://huggingface.co/spaces/{space_id}"
    
    print("步驟 2: Space 配置")
    print(f"Space ID: {space_id}")
    print(f"Space URL: {space_url}")
    print()
    
    # 3. 創建或更新 Space
    print("步驟 3: 創建/更新 Space")
    if not create_or_update_space(space_id):
        sys.exit(1)
    print()
    
    # 4. 準備檔案
    print("步驟 4: 準備檔案")
    temp_dir = Path(__file__).parent.parent / "temp_deploy"
    temp_dir.mkdir(exist_ok=True)
    
    if not prepare_files(temp_dir):
        sys.exit(1)
    print()
    
    # 5. 上傳
    print("步驟 5: 上傳檔案")
    success = upload_to_space(space_id, temp_dir)
    
    # 清理
    print("\n清理暫存檔案...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 結果
    print("\n" + "=" * 60)
    if success:
        print("✅ 部署成功!")
        print(f"\n🌐 訪問你的 Space: {space_url}")
        print("\n⏳ 注意: 首次部署需要 10-15 分鐘建置")
        print("   - Docker image 建置: 5-10 分鐘")
        print("   - 模型下載: 3-5 分鐘")
        print("\n💡 查看建置狀態:")
        print(f"   {space_url}/settings")
    else:
        print("❌ 部署失敗")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
