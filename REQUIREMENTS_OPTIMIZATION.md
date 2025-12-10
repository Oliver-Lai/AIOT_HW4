# 📦 Requirements 優化說明

## 原始大小 vs 優化後大小

### 原始 requirements.txt (完整開發環境)
```
tensorflow              ~500 MB
pandas                  ~15 MB
matplotlib              ~50 MB
seaborn                 ~10 MB
scikit-learn            ~30 MB
jupyter                 ~100 MB
pytest + 其他測試工具    ~20 MB
-----------------------------------
總計：                  ~725 MB
```

### 優化後 requirements.txt (僅 Streamlit 部署)
```
tensorflow-cpu          ~200 MB  (CPU 版本，無 GPU 支持)
streamlit               ~20 MB
streamlit-drawable-canvas ~1 MB
numpy                   ~15 MB
opencv-python-headless  ~40 MB   (無 GUI 依賴)
Pillow                  ~10 MB
-----------------------------------
總計：                  ~286 MB (減少 60%！)
```

## 移除的套件及原因

### ❌ 開發/測試工具 (不需要在生產環境)
- `jupyter` (100 MB) - 僅開發時使用
- `ipykernel` - Jupyter 依賴
- `pytest`, `pytest-cov` - 測試工具
- `black`, `flake8` - 代碼格式化工具

### ❌ 數據分析工具 (app.py 未使用)
- `pandas` (15 MB) - app.py 沒有使用
- `matplotlib` (50 MB) - 僅用於訓練/評估階段
- `seaborn` (10 MB) - 僅用於可視化
- `scikit-learn` (30 MB) - 僅用於訓練時的數據分割

### ❌ 其他工具
- `h5py` - 不需要，Keras 已內建支持
- `tqdm` - 不需要進度條

### ✅ 保留的核心套件
- `tensorflow-cpu` - 模型推論（使用 CPU 版本減少 60% 大小）
- `streamlit` - Web 框架
- `streamlit-drawable-canvas` - 繪圖畫布
- `numpy` - 數值計算
- `opencv-python-headless` - 圖像處理（無 GUI 版本）
- `Pillow` - 圖像處理

## 進一步優化建議

### 選項 1：使用 TensorFlow Lite (更小！)
如果願意轉換模型，可以使用 TFLite：
```
# requirements.txt
tensorflow-lite>=2.16.0  # 僅 ~5 MB！
streamlit>=1.25.0
streamlit-drawable-canvas>=0.9.0
numpy>=1.23.0
opencv-python-headless>=4.7.0
Pillow>=9.0.0
```
總大小：~91 MB (減少 87%！)

需要轉換模型：
```python
# 轉換腳本
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 選項 2：使用 ONNX Runtime (更快！)
```
# requirements.txt
onnxruntime>=1.16.0      # ~10 MB
streamlit>=1.25.0
streamlit-drawable-canvas>=0.9.0
numpy>=1.23.0
opencv-python-headless>=4.7.0
Pillow>=9.0.0
```
總大小：~96 MB

需要轉換模型到 ONNX 格式。

## 部署平台建議

### Streamlit Cloud
- **免費層級**: 1 GB RAM
- **推薦配置**: 當前優化版本 (~286 MB) ✅
- **狀態**: 應該可以運行

### Hugging Face Spaces
- **免費層級**: 16 GB RAM
- **推薦配置**: 原始或優化版本都可以
- **狀態**: 絕對沒問題 ✅

### Heroku
- **免費層級**: 已終止
- **付費層級**: 512 MB RAM (最低)
- **推薦配置**: 需要 TFLite 版本
- **狀態**: 不推薦

## 當前狀態

✅ **已優化 requirements.txt**
- 從 12 個套件減少到 6 個核心套件
- 預估大小從 ~725 MB 減少到 ~286 MB
- 減少 60% 安裝大小
- 保留所有 app.py 需要的功能

## 測試優化後的環境

```bash
# 創建新的虛擬環境測試
python -m venv test_env
source test_env/bin/activate

# 安裝優化後的依賴
pip install -r requirements.txt

# 測試應用
streamlit run app.py
```

如果一切正常，就可以部署了！
