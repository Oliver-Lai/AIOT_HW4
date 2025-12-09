# AIOT_HW4

EMNIST Handwritten Character Recognition System - A deep learning application for recognizing handwritten English letters and digits.

## Project Overview

This project implements an end-to-end character recognition system that:
1. Trains a CNN model on the EMNIST ByClass dataset (62 classes: 0-9, A-Z, a-z)
2. Deploys an interactive web interface using Streamlit
3. Allows users to draw characters and receive real-time predictions

## Features

- 🧠 **Deep Learning Model**: CNN trained on 697k+ EMNIST images
- 🎨 **Interactive Canvas**: Draw characters with your mouse/touch
- 📊 **Top-5 Predictions**: See confidence scores for multiple candidates
- ☁️ **Cloud Deployment**: Accessible via Streamlit Cloud
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile

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
│   └── utils/                 # Utility functions
├── app.py                     # Streamlit web application (coming soon)
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

**Training the Model** (coming soon)
```bash
python src/training/train.py
```

**Running the Web Application** (coming soon)
```bash
streamlit run app.py
```

## Development Status

🚧 **Phase 1: Project Setup & Environment** - In Progress

- [x] Initialize project structure
- [ ] Install dependencies
- [ ] Download EMNIST dataset
- [ ] Implement model training
- [ ] Create web interface
- [ ] Deploy to Streamlit Cloud

## Technical Stack

- **Framework**: TensorFlow/Keras
- **Web Interface**: Streamlit
- **Drawing Canvas**: streamlit-drawable-canvas
- **Image Processing**: OpenCV/Pillow
- **Data Science**: NumPy, Pandas, Matplotlib, scikit-learn
- **Deployment**: Streamlit Cloud

## Model Performance

Target metrics:
- **Accuracy**: ≥85% on EMNIST test set
- **Inference Time**: <2 seconds end-to-end
- **Model Size**: <100MB for cloud deployment

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