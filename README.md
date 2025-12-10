# AIOT_HW4 - EMNIST Handwritten Character Recognition System

> **Beyond MNIST**: This project uses **EMNIST ByClass dataset** to recognize not just handwritten digits (0-9), but also **uppercase letters (A-Z)** and **lowercase letters (a-z)** - a total of **62 character classes**!

## 🤖 Development Process with AI Assistance

This project was developed through an iterative conversation with an LLM assistant, following best practices in AI-assisted software development:

### Development Methodology
- **OpenSpec Framework**: Structured project planning with proposal, design, and task tracking
- **Phase-by-Phase Implementation**: 9 clearly defined phases from setup to deployment
- **Test-Driven Development**: 46 unit and integration tests ensuring code quality
- **Continuous Validation**: Each phase validated before proceeding to next

### Key Conversation Milestones
1. **Initial Setup (Phase 1)**: Project structure, dependencies, and environment configuration
2. **Data Exploration (Phase 2)**: EMNIST dataset analysis and visualization
3. **Preprocessing Pipeline (Phase 3)**: Memory-efficient data preprocessing with augmentation
4. **Model Development (Phase 4)**: CNN architecture design (1.7M parameters) and training
5. **Memory Optimization**: Solved RAM overflow issues with in-place preprocessing
6. **Model Evaluation (Phase 5)**: Comprehensive metrics, confusion matrix, per-class analysis
7. **Web Interface (Phase 6)**: Interactive Streamlit app with canvas drawing
8. **Bug Fixes**: Fixed image preprocessing (RGB vs alpha channel) and clear canvas functionality
9. **Testing & Validation (Phase 7)**: 46 tests covering all components (100% pass rate)
10. **Deployment Preparation (Phase 8)**: Streamlit Cloud configuration and documentation
11. **Final Documentation (Phase 9)**: Complete project handoff with technical docs

### Development Statistics
- **Total Phases**: 9
- **Tests Written**: 46 (100% passing)
- **Lines of Code**: ~3,500+
- **Model Size**: 20.5 MB
- **Development Time**: ~12 hours of LLM interaction
- **Iterations**: Multiple refinements per phase

## Project Overview

This project implements an **end-to-end character recognition system** that goes far beyond traditional digit recognition:

### 🎯 What Makes This Special?
Unlike MNIST (10 digit classes), this system uses **EMNIST ByClass** to recognize:
- ✅ **Digits**: 0-9 (10 classes)
- ✅ **Uppercase Letters**: A-Z (26 classes)
- ✅ **Lowercase Letters**: a-z (26 classes)
- 🎉 **Total**: **62 character classes**!

### System Capabilities
1. **Train a CNN model** on 697,932 EMNIST training images
2. **Deploy an interactive web interface** using Streamlit
3. **Real-time character recognition** with confidence scores
4. **Top-5 predictions** to show alternative interpretations
5. **Cross-browser support** including mobile devices

## ✨ Features

### 🎯 Beyond MNIST - Full Alphabet Recognition
- **62 Character Classes**: Digits (0-9) + Uppercase (A-Z) + Lowercase (a-z)
- **697,932 Training Images**: Large-scale EMNIST ByClass dataset
- **116,323 Test Images**: Comprehensive evaluation dataset
- **Real-World Applicability**: Recognize any handwritten alphanumeric character

### 🧠 Advanced Deep Learning
- **CNN Architecture**: 1.7M parameters with 3 convolutional blocks
- **Batch Normalization**: Improved training stability
- **Dropout Regularization**: Prevent overfitting
- **Data Augmentation**: Rotation, shifts, zoom for robustness
- **≥85% Accuracy**: High performance on test set

### 🎨 Interactive User Experience
- **Drawing Canvas**: 280×280 pixel canvas with adjustable stroke width
- **Real-time Prediction**: Results in <100ms (warm)
- **Top-5 Confidence Scores**: See alternative interpretations
- **Low Confidence Warnings**: Alert when prediction is uncertain
- **Clear & Retry**: Easy canvas reset for multiple attempts

### ☁️ Production-Ready Deployment
- **Streamlit Cloud**: One-click deployment
- **20.5 MB Model**: Fast loading, under 100MB limit
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Cross-Browser**: Chrome, Firefox, Safari, Edge
- **Touch Support**: Draw with finger on mobile devices

### 🧪 Quality Assurance
- **46 Tests**: 100% passing (unit + integration)
- **Memory Optimized**: Handles large datasets efficiently
- **Error Handling**: Graceful handling of edge cases
- **Performance Tested**: <50ms inference time

## Project Structure

```
AIOT_HW4/
├── data/                      # Dataset storage (gitignored)
│   └── emnist/                # EMNIST downloaded data
├── models/                    # Saved trained models (gitignored)
│   └── .gitkeep
├── notebooks/                 # Jupyter notebooks for development
├── src/                       # Source code
│   ├── data/                  # Dataset loading and preprocessing
│   ├── models/                # Model architectures
│   ├── training/              # Training scripts
│   ├── preprocessing/         # Data preprocessing functions
│   └── utils/                 # Utility functions
├── app.py                     # Streamlit web application ✅
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Oliver-Lai/AIOT_HW4.git
   cd AIOT_HW4
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv venv
   
   # Activate on Linux/Mac
   source venv/bin/activate
   
   # Activate on Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import tensorflow; import streamlit; print('✓ All dependencies installed')"
   ```

### Usage

#### 1. Download and Prepare Data
```bash
python src/data/dataset.py
```

#### 2. Train the Model (optional - model already provided)
```bash
# Train with full dataset (requires ~8-10GB RAM)
python src/training/train.py

# Or train with subset for faster training
python train_quick_model.py
```

#### 3. Run the Web Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Draw a character on the canvas and click "Predict" to see the results!

#### 4. Evaluate Model Performance
```bash
# Run evaluation script
python src/training/evaluate.py --model models/emnist_cnn_v1.keras

# Or use the evaluation notebook
jupyter notebook notebooks/04_model_evaluation.ipynb
```

## Development Status

✅ **Phase 1: Project Setup & Environment** - Complete
✅ **Phase 2: Data Acquisition & Exploration** - Complete
✅ **Phase 3: Data Preprocessing Pipeline** - Complete
✅ **Phase 4: Model Development** - Complete
✅ **Phase 5: Model Evaluation** - Complete
✅ **Phase 6: Streamlit Application Development** - Complete

Progress: 6/9 phases complete (66.7%)

- [x] Initialize project structure
- [x] Install dependencies
- [x] Download EMNIST dataset
- [x] Implement preprocessing pipeline
- [x] Build CNN model (1.7M parameters)
- [x] Train model (achieved ~85% accuracy)
- [x] Create evaluation tools
- [x] Build interactive web interface
- [ ] Write comprehensive tests (Phase 7)
- [ ] Deploy to Streamlit Cloud (Phase 8)
- [ ] Complete documentation (Phase 9)

## Technical Stack

- **Framework**: TensorFlow/Keras
- **Web Interface**: Streamlit
- **Drawing Canvas**: streamlit-drawable-canvas
- **Image Processing**: OpenCV/Pillow
- **Data Science**: NumPy, Pandas, Matplotlib, scikit-learn
- **Deployment**: Streamlit Cloud

## 📊 Model Performance

### Achieved Metrics
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Accuracy** | ≥85% | ~85%+ | ✅ |
| **Model Size** | <100 MB | 20.5 MB | ✅ |
| **Parameters** | 1-3M | 1.7M | ✅ |
| **Cold Start** | <7s | ~5s | ✅ |
| **Warm Inference** | <100ms | 49.7ms | ✅ |
| **End-to-End** | <2s | <1s | ✅ |

### Dataset Scale
- **Training Set**: 697,932 images
- **Test Set**: 116,323 images
- **Classes**: 62 (10 digits + 52 letters)
- **Image Size**: 28×28 pixels
- **Format**: Grayscale, normalized [0, 1]

### Character Recognition Breakdown
- **Digits (0-9)**: 10 classes - Numbers recognition
- **Uppercase (A-Z)**: 26 classes - Capital letters
- **Lowercase (a-z)**: 26 classes - Small letters
- **Total Diversity**: 62 unique characters vs MNIST's 10 digits!

## Documentation

See `openspec/changes/add-emnist-recognition-system/` for detailed:
- **proposal.md** - Project overview, goals, and success criteria
- **design.md** - Architecture, tech stack, and technical decisions
- **tasks.md** - Implementation tasks and timeline
- **specs/** - Detailed capability specifications

## License

This project is created for educational purposes.

## Contributing

This is a homework project. For questions or suggestions, please open an issue.

## Acknowledgments

- EMNIST Dataset: Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017)
- Streamlit framework for rapid ML application development