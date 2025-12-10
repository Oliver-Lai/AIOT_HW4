# AIOT_HW4 - EMNIST Handwritten Character Recognition

> **Beyond MNIST**: Recognizes **62 character classes** - digits (0-9), uppercase (A-Z), and lowercase (a-z) letters!

## 🎯 Features

- ✍️ **Interactive Drawing Canvas** - Draw characters directly in your browser
- 🧠 **Deep CNN Model** - 1.7M parameters, 85%+ accuracy on EMNIST ByClass
- ⚡ **Real-time Predictions** - Sub-100ms inference time
- 📊 **Top-5 Results** - See confidence scores for multiple predictions
- 🎨 **62 Character Classes** - 6.2× more capable than traditional MNIST

## 🚀 Quick Start

### Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Deploy to Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy with `app.py` as the main file

## 📊 Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| **Accuracy** | ~85%+ | ✅ |
| **Model Size** | 20.5 MB | ✅ |
| **Parameters** | 1.7M | ✅ |
| **Inference Time** | ~50ms | ✅ |
| **Character Classes** | 62 | ✅ |

### Character Recognition Capabilities
- **Digits (0-9)**: 10 classes
- **Uppercase (A-Z)**: 26 classes  
- **Lowercase (a-z)**: 26 classes
- **Total**: 62 unique characters (vs MNIST's 10 digits)

## 🏗️ Project Structure

```
AIOT_HW4/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies (optimized)
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── models/
│   ├── emnist_cnn_v1.keras    # Trained CNN model (20.5 MB)
│   └── label_mapping.json     # Character class mappings
├── openspec/                   # Project documentation
└── README.md                   # This file
```

## 🛠️ Technical Stack

- **Framework**: TensorFlow 2.16+ (CPU optimized)
- **Web Interface**: Streamlit
- **Drawing Component**: streamlit-drawable-canvas
- **Image Processing**: OpenCV (headless), Pillow
- **Model**: Custom CNN with 3 convolutional blocks + 2 dense layers

## 🎓 Model Architecture

```
Input (28×28×1)
    ↓
Conv2D(64) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D(128) + BatchNorm + ReLU + MaxPool
    ↓
Conv2D(256) + BatchNorm + ReLU + MaxPool
    ↓
Flatten → Dense(512) + Dropout(0.5)
    ↓
Dense(256) + Dropout(0.3)
    ↓
Dense(62) + Softmax
    ↓
Output (62 classes)
```

## 🤖 Development with AI

This project was developed using AI-assisted development methodology:
- **OpenSpec Framework**: Structured planning and documentation
- **9 Development Phases**: From setup to deployment
- **Iterative Refinement**: Bug fixes and optimizations throughout
- **Complete Documentation**: See `openspec/` directory for details

## 📝 Usage

1. **Draw**: Use your mouse or touchscreen to draw a character
2. **Predict**: Click the predict button or it predicts automatically
3. **View Results**: See top-5 predictions with confidence scores
4. **Clear**: Reset the canvas to try another character

### Tips for Best Results
- Draw characters clearly and centered
- Use the full canvas space
- Try different stroke widths for better recognition
- Capital vs lowercase matters!

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- **EMNIST Dataset**: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017)
- **Streamlit**: For the excellent web framework
- **TensorFlow/Keras**: For the deep learning framework

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Model Accuracy**: ~85%+  
**Deployment**: Optimized for Streamlit Cloud
