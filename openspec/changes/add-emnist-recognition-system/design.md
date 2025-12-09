# Design: EMNIST Handwritten Character Recognition System

## Architecture Overview

### System Components
```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web App                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │   Drawing    │  │    Image     │  │   Prediction    │   │
│  │   Canvas     │→ │ Preprocessing│→ │    Display      │   │
│  │  Component   │  │   Pipeline   │  │   Component     │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│         ↓                  ↓                    ↑            │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │  UI State    │  │  Trained CNN │  │   Top-K         │   │
│  │  Manager     │  │    Model     │→ │ Predictions     │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Model Training Pipeline (Offline)               │
├─────────────────────────────────────────────────────────────┤
│  EMNIST Dataset → Preprocessing → CNN Training → Evaluation │
│                                          ↓                    │
│                                   Saved Model (.h5/.pt)      │
└─────────────────────────────────────────────────────────────┘
```

### Component Interactions
1. **User draws** on canvas → Raw image data captured
2. **Image preprocessing** → Resize, normalize, reshape to match training format
3. **Model inference** → CNN predicts character probabilities
4. **Display results** → Show top predictions with confidence scores
5. **User clears canvas** → Reset state for new input

## Technology Stack

### Model Training
- **Framework**: TensorFlow/Keras (preferred for Streamlit Cloud compatibility)
  - Alternative: PyTorch (requires ONNX conversion for deployment)
- **Dataset**: EMNIST ByClass (62 classes: 10 digits + 26 uppercase + 26 lowercase)
- **Model Architecture**: Convolutional Neural Network (CNN)
  - Conv2D layers for feature extraction
  - MaxPooling for dimensionality reduction
  - Dense layers for classification
  - Dropout for regularization
- **Training Tools**: 
  - NumPy for data manipulation
  - Matplotlib/Seaborn for visualization
  - scikit-learn for metrics

### Web Application
- **Frontend Framework**: Streamlit
- **Drawing Component**: streamlit-drawable-canvas
- **Image Processing**: OpenCV (cv2) or Pillow (PIL)
- **Model Loading**: TensorFlow/Keras native loading or ONNX Runtime
- **Deployment**: Streamlit Cloud

### File Structure
```
AIOT_HW4/
├── data/                      # Dataset storage
│   └── emnist/                # EMNIST downloaded data
├── models/                    # Saved trained models
│   ├── emnist_cnn_v1.h5      # Keras model
│   └── label_mapping.json     # Class index to character mapping
├── notebooks/                 # Development notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
├── src/                       # Source code
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py         # EMNIST loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py             # Model architecture
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py           # Training script
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py   # Image preprocessing utilities
├── app.py                     # Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## Data Flow

### Training Phase
1. **Download EMNIST**: Use torchvision.datasets or tensorflow_datasets
2. **Preprocess**: 
   - Normalize pixel values (0-255 → 0-1)
   - Reshape to (28, 28, 1) for grayscale
   - One-hot encode labels
3. **Augmentation**: Random rotation, slight shifts for robustness
4. **Training**: 
   - Train/validation/test split (70/15/15)
   - Batch size: 128
   - Optimizer: Adam
   - Loss: Categorical crossentropy
   - Early stopping based on validation loss
5. **Save**: Export model and label mapping

### Inference Phase
1. **Capture**: Get image from drawable canvas (RGBA format)
2. **Preprocess**:
   - Convert RGBA → grayscale
   - Resize to 28x28 pixels
   - Invert colors if needed (EMNIST has white on black)
   - Normalize pixel values
   - Reshape to (1, 28, 28, 1) for batch inference
3. **Predict**: Model outputs 62-element probability vector
4. **Post-process**: 
   - Get top-K predictions (K=5)
   - Map indices to characters using label_mapping.json
   - Format with confidence percentages
5. **Display**: Show results in Streamlit UI

## Model Architecture

### CNN Design
```python
Input: (28, 28, 1)
    ↓
Conv2D(32, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
    ↓
Conv2D(64, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
    ↓
Conv2D(128, 3x3, ReLU) → BatchNorm → MaxPool(2x2)
    ↓
Flatten
    ↓
Dense(256, ReLU) → Dropout(0.5)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(62, Softmax)
    ↓
Output: 62 class probabilities
```

### Rationale
- **3 Convolutional blocks**: Progressive feature extraction from edges to complex patterns
- **Batch Normalization**: Stabilize training and improve convergence
- **Dropout**: Prevent overfitting on training data
- **Architecture size**: Balances accuracy with inference speed for web deployment

## Streamlit UI Design

### Layout
```
┌─────────────────────────────────────────────────────────┐
│  EMNIST Handwritten Character Recognition               │
│  ─────────────────────────────────────────────────      │
│                                                          │
│  Instructions: Draw a character in the canvas below     │
│                                                          │
│  ┌────────────────────────┐  ┌──────────────────────┐  │
│  │                        │  │  Predictions:        │  │
│  │   Drawing Canvas       │  │                      │  │
│  │   (280x280 pixels)     │  │  1. A (95.2%)       │  │
│  │                        │  │  2. H (2.3%)        │  │
│  │                        │  │  3. R (1.1%)        │  │
│  │                        │  │  4. K (0.8%)        │  │
│  │                        │  │  5. N (0.6%)        │  │
│  └────────────────────────┘  └──────────────────────┘  │
│                                                          │
│  [Clear Canvas]  [Predict]                              │
│                                                          │
│  ─────────────────────────────────────────────────      │
│  Model Info: EMNIST ByClass | 62 Classes | 95.3% Acc   │
└─────────────────────────────────────────────────────────┘
```

### Interaction Flow
1. User draws character on canvas
2. Click "Predict" button → triggers inference
3. Results update in sidebar/column
4. Click "Clear Canvas" → reset for new character
5. Optional: "About" expander with model details

## Preprocessing Pipeline

### Critical Considerations
EMNIST images have specific characteristics:
- **Resolution**: 28x28 pixels
- **Color**: Grayscale (1 channel)
- **Orientation**: Characters are white on black background
- **Normalization**: Pixels in range [0, 1]

User canvas drawings need transformation:
```python
def preprocess_canvas_image(image_data):
    """
    Transform canvas drawing to EMNIST format
    
    Args:
        image_data: RGBA image from canvas (variable size)
    
    Returns:
        Preprocessed image ready for model (1, 28, 28, 1)
    """
    # Step 1: Extract alpha channel (drawing data)
    img = image_data[:, :, 3]  # Alpha channel contains drawing
    
    # Step 2: Resize to 28x28
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Step 3: Invert (canvas is black on white, EMNIST is white on black)
    img = 255 - img
    
    # Step 4: Normalize to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Step 5: Reshape for model input
    img = img.reshape(1, 28, 28, 1)
    
    return img
```

## Deployment Strategy

### Streamlit Cloud Requirements
1. **Repository**: GitHub repo with all code
2. **requirements.txt**: All Python dependencies with versions
3. **app.py**: Main Streamlit application entry point
4. **Model files**: Include pre-trained model in repo or download on startup
5. **Configuration**: `.streamlit/config.toml` for theme/settings

### Optimization for Cloud Deployment
- **Model size**: Keep under 100MB if possible
  - Use `model.save()` with compression
  - Consider model quantization for size reduction
- **Memory**: Streamlit Cloud has 1GB RAM limit
  - Lazy load model (only when needed)
  - Clear cache between predictions if needed
- **Cold start**: First load may be slow
  - Pre-load model at app startup
  - Add loading spinner with status

### Environment Variables
```toml
# .streamlit/config.toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 5
enableXsrfProtection = true
```

## Error Handling

### Potential Issues & Solutions
1. **Empty canvas**: Validate image has content before prediction
2. **Model loading failure**: Show error message with troubleshooting steps
3. **Preprocessing errors**: Catch and log with helpful user message
4. **Low confidence predictions**: Display warning if top prediction < 50%
5. **Canvas browser compatibility**: Test on Chrome, Firefox, Safari

## Testing Strategy

### Model Testing
- Unit tests for preprocessing functions
- Integration tests for end-to-end prediction pipeline
- Validation on EMNIST test set
- Manual testing with hand-drawn samples

### Application Testing
- UI interaction testing (draw, predict, clear)
- Cross-browser testing
- Mobile responsiveness check
- Performance testing (inference time < 2s)

## Security & Privacy
- No user data persistence (drawings not saved)
- Model runs client-side (no data sent to external APIs)
- Standard Streamlit Cloud security practices
- No authentication required (public demo)

## Future Enhancements (Out of Scope)
- Support for multiple character sequences (word recognition)
- Training data contribution from users
- Model retraining with user feedback
- Mobile app version
- Advanced architectures (ResNet, EfficientNet)
- Multi-language support
