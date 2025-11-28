"""
自動部署到 Hugging Face Spaces
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
        print("\n請執行以下步驟:")
        print("1. 訪問 https://huggingface.co/settings/tokens")
        print("2. 創建新的 Access Token (需要 'write' 權限)")
        print("3. 執行: huggingface-cli login")
        print("   或: python -c \"from huggingface_hub import login; login()\"")
        sys.exit(1)


def create_space(space_id):
    """創建 Hugging Face Space"""
    try:
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True
        )
        print(f"✓ Space 創建成功: {space_id}")
        return True
    except Exception as e:
        print(f"❌ 創建 Space 失敗: {e}")
        return False


def prepare_files(temp_dir):
    """準備要上傳的檔案"""
    print("\n準備檔案...")
    
    # 必要檔案列表
    files_to_copy = [
        'app.py',
        'model_utils.py',
        'tts_utils.py',
        'requirements.txt',
        'Dockerfile'
    ]
    
    # 複製檔案
    for file in files_to_copy:
        src = Path(file)
        if src.exists():
            dst = temp_dir / file
            shutil.copy2(src, dst)
            print(f"  ✓ {file}")
        else:
            print(f"  ⚠️  {file} 不存在")
    
    # 使用 HF 專用 README
    readme_src = Path('README_HF.md')
    if readme_src.exists():
        shutil.copy2(readme_src, temp_dir / 'README.md')
        print(f"  ✓ README.md (from README_HF.md)")
    else:
        readme_src = Path('README.md')
        if readme_src.exists():
            shutil.copy2(readme_src, temp_dir / 'README.md')
            print(f"  ✓ README.md")
    
    # 創建 .gitignore
    gitignore_content = """__pycache__/
*.pyc
*.pyo
models/
temp_audio/
*.mp3
*.wav
.streamlit/secrets.toml
.env
"""
    with open(temp_dir / '.gitignore', 'w') as f:
        f.write(gitignore_content)
    print(f"  ✓ .gitignore")


def upload_to_space(space_id, temp_dir):
    """上傳檔案到 Space"""
    print(f"\n上傳到 Space: {space_id}")
    
    api = HfApi()
    
    try:
        api.upload_folder(
            folder_path=str(temp_dir),
            repo_id=space_id,
            repo_type="space",
            commit_message="Deploy Chinglish Generator to Hugging Face Spaces"
        )
        print("✓ 上傳成功")
        return True
    except Exception as e:
        print(f"❌ 上傳失敗: {e}")
        return False


def main():
    print("=" * 60)
    print("晶晶體生成器 - Hugging Face Spaces 部署工具")
    print("=" * 60)
    print()
    
    # 1. 檢查登入
    print("步驟 1: 檢查登入狀態")
    username = check_login()
    print()
    
    # 2. 輸入 Space 名稱
    print("步驟 2: 配置 Space")
    space_name = input("請輸入 Space 名稱 (預設: chinglish-generator): ").strip()
    if not space_name:
        space_name = "chinglish-generator"
    
    space_id = f"{username}/{space_name}"
    space_url = f"https://huggingface.co/spaces/{space_id}"
    
    print(f"\nSpace ID: {space_id}")
    print(f"Space URL: {space_url}")
    print()
    
    # 確認
    confirm = input("繼續部署? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("已取消")
        return
    
    # 3. 創建 Space
    print("\n步驟 3: 創建 Space")
    if not create_space(space_id):
        return
    
    # 4. 準備檔案
    print("\n步驟 4: 準備檔案")
    temp_dir = Path('temp_hf_deploy')
    temp_dir.mkdir(exist_ok=True)
    
    try:
        prepare_files(temp_dir)
        
        # 5. 上傳
        print("\n步驟 5: 上傳檔案")
        if not upload_to_space(space_id, temp_dir):
            return
        
    finally:
        # 清理臨時目錄
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("\n✓ 已清理臨時檔案")
    
    # 完成
    print("\n" + "=" * 60)
    print("🎉 部署成功！")
    print("=" * 60)
    print()
    print("你的 Space 正在構建中...")
    print()
    print(f"Space URL: {space_url}")
    print()
    print("首次啟動需要 5-10 分鐘來:")
    print("  1. 安裝依賴套件")
    print("  2. 下載模型 (約 1GB)")
    print("  3. 啟動應用程式")
    print()
    print(f"查看構建日誌: {space_url}/logs")
    print()
    print("提示:")
    print("  - Space 會在 48 小時無使用後進入睡眠")
    print("  - 首次訪問會喚醒 (約 30 秒)")
    print("  - 可在 Settings 中配置硬體 (免費: CPU basic)")
    print()
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已中斷")
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
