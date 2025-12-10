# Streamlit Cloud 部署說明

## 問題排查

如果遇到 "pyproject.toml" 相關錯誤或部署失敗，請檢查：

### ✅ 已完成的優化

1. **簡化 requirements.txt**
   - 移除了所有非必要套件（pandas, matplotlib, jupyter 等）
   - 使用 `tensorflow-cpu` 而非完整版 tensorflow
   - 只保留 6 個核心套件
   - 總大小：~286 MB（符合 Streamlit Cloud 限制）

2. **優化 .streamlit/config.toml**
   - 移除了可能衝突的配置（port, enableCORS）
   - 設置 `maxUploadSize = 50` MB
   - 保留基本主題和服務器設置

3. **確保 .gitignore 正確**
   - 排除 `data/` 目錄（包含 197 MB 數據集）
   - 保留 `models/emnist_cnn_v1.keras`（20.5 MB，需要用於推論）
   - 排除所有 Python 緩存和虛擬環境

### 📦 Streamlit Cloud 部署文件清單

**必須上傳的文件：**
```
✅ app.py                          # 主應用（311 行）
✅ requirements.txt                # 優化後的依賴（6 個套件）
✅ .streamlit/config.toml          # Streamlit 配置
✅ models/emnist_cnn_v1.keras      # 訓練好的模型（20.5 MB）
✅ models/label_mapping.json       # 字符映射（<1 KB）
```

**不應上傳的文件：**
```
❌ data/                          # 197 MB 數據集（訓練時用，部署不需要）
❌ notebooks/                     # Jupyter notebooks（開發用）
❌ tests/                         # 測試文件（CI/CD 用）
❌ .venv/                         # 虛擬環境
❌ __pycache__/                   # Python 緩存
❌ *.pyc                          # 編譯的 Python 文件
```

### 🚀 Streamlit Cloud 部署步驟

#### 方法 1：從 GitHub 部署（推薦）

1. **確保 Git 倉庫乾淨**
   ```bash
   # 檢查哪些文件會被上傳
   git status
   
   # 清理不需要的文件
   git clean -fdX  # 清理 .gitignore 中的文件
   ```

2. **推送到 GitHub**
   ```bash
   git add .
   git commit -m "Optimize for Streamlit Cloud deployment"
   git push origin main
   ```

3. **在 Streamlit Cloud 部署**
   - 訪問：https://share.streamlit.io/
   - 點擊 "New app"
   - 選擇你的 GitHub 倉庫：`Oliver-Lai/AIOT_HW4`
   - Main file path: `app.py`
   - 點擊 "Deploy"

#### 方法 2：直接從本地部署

如果 GitHub 上傳太慢或有問題：

1. **創建最小化的部署包**
   ```bash
   mkdir streamlit_deploy
   cp app.py streamlit_deploy/
   cp requirements.txt streamlit_deploy/
   cp -r .streamlit streamlit_deploy/
   cp -r models streamlit_deploy/
   ```

2. **初始化新的 Git 倉庫**
   ```bash
   cd streamlit_deploy
   git init
   git add .
   git commit -m "Minimal Streamlit deployment"
   ```

3. **推送到新的 GitHub 倉庫並部署**

### 🔧 常見部署錯誤解決

#### 錯誤 1: "ModuleNotFoundError: No module named 'XXX'"
**原因**: requirements.txt 缺少必要的套件  
**解決**: 確保 requirements.txt 包含所有 app.py 使用的套件

目前的 requirements.txt 已包含：
- tensorflow-cpu
- streamlit
- streamlit-drawable-canvas
- numpy
- opencv-python-headless
- Pillow

#### 錯誤 2: "Memory limit exceeded"
**原因**: 應用使用超過 1 GB RAM  
**解決**: 
- ✅ 已使用 `tensorflow-cpu`（較小）
- ✅ 已移除不必要的套件
- ✅ 使用 `@st.cache_resource` 緩存模型

#### 錯誤 3: "App failed to load"
**原因**: 模型文件未正確上傳  
**解決**: 
```bash
# 檢查模型文件是否存在且大小正確
ls -lh models/emnist_cnn_v1.keras
# 應顯示 ~20.5 MB
```

#### 錯誤 4: "Build failed" 或 pyproject.toml 錯誤
**原因**: Streamlit Cloud 嘗試安裝開發依賴  
**解決**:
- ✅ 確保沒有 `pyproject.toml`（專案中已無此文件）
- ✅ 確保沒有 `setup.py`
- ✅ 只使用 `requirements.txt`

### 📊 預期資源使用

部署到 Streamlit Cloud 後的預期資源使用：

```
安裝大小：     ~286 MB
運行時內存：   ~400-500 MB（包含模型）
冷啟動時間：   30-60 秒（首次加載）
熱啟動時間：   <5 秒（已緩存模型）
推論時間：     50-100 ms
```

**結論**: 應該可以在 Streamlit Cloud 免費層級（1 GB RAM）順利運行！✅

### 🎯 部署檢查清單

在部署前確認：

- [x] requirements.txt 已優化（只有 6 個套件）
- [x] .streamlit/config.toml 已簡化
- [x] .gitignore 正確排除大文件
- [x] 模型文件 (20.5 MB) 存在
- [x] app.py 沒有使用已移除的套件
- [x] 沒有 pyproject.toml 或 setup.py

全部完成！可以開始部署了。

### 📞 如果還是失敗

1. **檢查 Streamlit Cloud 日誌**
   - 部署頁面的 "Manage app" → "Logs"
   - 查看具體錯誤信息

2. **本地測試**
   ```bash
   # 創建新的虛擬環境
   python -m venv test_deploy
   source test_deploy/bin/activate
   
   # 安裝優化後的依賴
   pip install -r requirements.txt
   
   # 測試應用
   streamlit run app.py
   ```

3. **聯繫支持**
   - Streamlit Community: https://discuss.streamlit.io/
   - 提供錯誤日誌和 requirements.txt

---

**當前狀態**: ✅ 已優化完成，可以部署！
