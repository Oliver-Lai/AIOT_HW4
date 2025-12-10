# Technical Documentation - EMNIST Character Recognition System

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface (Streamlit)                │
│  - Drawing Canvas (280×280)                                  │
│  - Prediction Display                                        │
│  - Results Visualization                                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Preprocessing Pipeline                          │
│  1. RGB to Grayscale Conversion                             │
│  2. Resize to 28×28                                         │
│  3. Color Inversion (black-on-white → white-on-black)       │
│  4. Normalization [0, 1]                                    │
│  5. Reshape to (1, 28, 28, 1)                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│            CNN Model (1.7M Parameters)                       │
│  - Input Layer: (28, 28, 1)                                 │
│  - Conv Block 1: 64 filters                                 │
│  - Conv Block 2: 128 filters                                │
│  - Conv Block 3: 256 filters                                │
│  - Dense Layer 1: 512 units                                 │
│  - Dense Layer 2: 256 units                                 │
│  - Output Layer: 62 classes (softmax)                       │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              Prediction Results                              │
│  - Top-5 Character Predictions                              │
│  - Confidence Scores                                         │
│  - Character Mapping                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### CNN Design (1,701,950 parameters)

#### Layer-by-Layer Breakdown

**Input Layer**
- Shape: (None, 28, 28, 1)
- Format: Grayscale images, normalized [0, 1]

**Convolutional Block 1**
```python
Conv2D(64, (3,3), padding='same')      # 64 filters, 3×3 kernel
BatchNormalization()                    # Normalize activations
Activation('relu')                      # Non-linear activation
MaxPooling2D((2,2))                    # Downsample to 14×14
```

**Convolutional Block 2**
```python
Conv2D(128, (3,3), padding='same')     # 128 filters
BatchNormalization()
Activation('relu')
MaxPooling2D((2,2))                    # Downsample to 7×7
```

**Convolutional Block 3**
```python
Conv2D(256, (3,3), padding='same')     # 256 filters
BatchNormalization()
Activation('relu')
MaxPooling2D((2,2))                    # Downsample to 3×3
```

**Dense Layers**
```python
Flatten()                               # Flatten to 1D: 2304 units
Dense(512)                              # Fully connected: 512 units
BatchNormalization()
Activation('relu')
Dropout(0.5)                           # 50% dropout for regularization

Dense(256)                              # Fully connected: 256 units
BatchNormalization()
Activation('relu')
Dropout(0.3)                           # 30% dropout

Dense(62, activation='softmax')         # Output: 62 classes
```

#### Parameter Distribution
- **Trainable Parameters**: 1,699,518
- **Non-trainable Parameters**: 2,432 (BatchNorm statistics)
- **Total Parameters**: 1,701,950

---

## Data Preprocessing Pipeline

### Training Data Preprocessing

#### 1. Normalization
```python
def normalize_images(images):
    """Scale pixel values from [0, 255] to [0, 1]"""
    return images.astype(np.float32) / 255.0
```

#### 2. Reshaping
```python
def reshape_images(images):
    """Add channel dimension: (n, 28, 28) → (n, 28, 28, 1)"""
    return images.reshape(-1, 28, 28, 1)
```

#### 3. Label Encoding
```python
def one_hot_encode_labels(labels, num_classes=62):
    """Convert integer labels to one-hot vectors"""
    return to_categorical(labels, num_classes)
```

#### 4. Train/Val Split
```python
def create_train_val_split(x, y, val_size=0.15):
    """Split data with stratification to maintain class balance"""
    return train_test_split(x, y, test_size=val_size, 
                           stratify=y, random_state=42)
```

#### 5. Data Augmentation
```python
ImageDataGenerator(
    rotation_range=15,          # Rotate ±15 degrees
    width_shift_range=0.1,      # Shift horizontally ±10%
    height_shift_range=0.1,     # Shift vertically ±10%
    zoom_range=0.1,             # Zoom ±10%
    fill_mode='nearest'         # Fill empty pixels
)
```

### Canvas Input Preprocessing

#### Step-by-Step Process

**Step 1: Extract RGB Channels**
```python
img_rgb = canvas_data[:, :, :3]  # Ignore alpha channel
```

**Step 2: Convert to Grayscale**
```python
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
```

**Step 3: Detect Empty Canvas**
```python
if img_gray.min() > 250:  # All white = empty
    return None
```

**Step 4: Resize to Model Input Size**
```python
img_resized = cv2.resize(img_gray, (28, 28), 
                        interpolation=cv2.INTER_AREA)
```

**Step 5: Invert Colors**
```python
# Canvas: black-on-white
# EMNIST: white-on-black
img_inverted = 255 - img_resized
```

**Step 6: Normalize**
```python
img_normalized = img_inverted.astype(np.float32) / 255.0
```

**Step 7: Reshape for Model**
```python
img_final = img_normalized.reshape(1, 28, 28, 1)
```

---

## Training Process

### Configuration

```python
TRAINING_CONFIG = {
    'batch_size': 128,
    'epochs': 25,
    'learning_rate': 0.001,
    'optimizer': 'Adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy', 'top_k_categorical_accuracy'],
    'validation_split': 0.15
}
```

### Memory Optimization

Due to RAM constraints with the full 697K dataset:

```python
# In-place preprocessing to avoid data copying
x_train = x_train.astype(np.float32)  # Convert in-place
x_train /= 255.0                      # Normalize in-place
x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape

# Subset training for faster iteration
SUBSET_SIZE = 100000  # Use 100K samples
indices = stratified_sample(y_train, SUBSET_SIZE)
x_train_subset = x_train[indices]
y_train_subset = y_train[indices]
```

### Callbacks

```python
callbacks = [
    ModelCheckpoint(
        'models/emnist_cnn_best.keras',
        monitor='val_accuracy',
        save_best_only=True
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )
]
```

---

## Prediction Pipeline

### Real-time Inference Flow

```python
def predict_character(model, canvas_image, label_mapping, top_k=5):
    """
    Complete prediction pipeline
    
    Args:
        model: Loaded Keras model
        canvas_image: Preprocessed image (1, 28, 28, 1)
        label_mapping: Dict mapping indices to characters
        top_k: Number of top predictions to return
    
    Returns:
        List of (character, confidence) tuples
    """
    # 1. Get model predictions (62 probabilities)
    predictions = model.predict(canvas_image, verbose=0)[0]
    
    # 2. Get top-k indices
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    
    # 3. Map to characters and format
    results = []
    for idx in top_indices:
        char = label_mapping[str(idx)]
        confidence = float(predictions[idx]) * 100
        results.append((char, confidence))
    
    return results
```

### Model Caching

```python
@st.cache_resource
def load_model():
    """Load model once and cache in memory"""
    return tf.keras.models.load_model('models/emnist_cnn_v1.keras')
```

---

## Performance Optimization

### 1. Model Loading
- **Caching**: `@st.cache_resource` ensures model loads only once
- **Lazy Loading**: Model loads on first prediction, not on app start
- **Size Optimization**: 20.5 MB model (vs potential 100+ MB)

### 2. Inference Speed
- **Warm Prediction**: 49.7 ms average
- **Batch Processing**: Not needed for single character prediction
- **TensorFlow Optimization**: Uses compiled graph after first call

### 3. Memory Management
- **In-place Operations**: Avoid data copying during preprocessing
- **Garbage Collection**: Explicit `gc.collect()` after training
- **Subset Training**: Use stratified samples for faster iteration

### 4. UI Responsiveness
- **Dynamic Canvas Key**: Session state for instant canvas clear
- **Spinners**: Show loading status during model operations
- **Error Handling**: Graceful degradation on failures

---

## Label Mapping

### EMNIST ByClass Character Order

```python
LABEL_MAPPING = {
    # Digits: 0-9 (indices 0-9)
    '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
    '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    
    # Uppercase: A-Z (indices 10-35)
    '10': 'A', '11': 'B', '12': 'C', ..., '35': 'Z',
    
    # Lowercase: a-z (indices 36-61)
    '36': 'a', '37': 'b', '38': 'c', ..., '61': 'z'
}
```

Total: 62 classes (10 + 26 + 26)

---

## Error Handling

### Canvas Preprocessing Errors
```python
# Empty canvas detection
if img_gray.min() > 250:
    st.warning("Canvas is empty! Please draw a character.")
    return None

# None input handling
if canvas_data is None:
    return None
```

### Model Loading Errors
```python
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please train a model first.")
    return None
```

### Prediction Errors
```python
try:
    predictions = model.predict(image, verbose=0)
except Exception as e:
    st.error(f"Prediction error: {e}")
    return None
```

---

## Testing Strategy

### Test Coverage

1. **Unit Tests** (12 preprocessing + 13 model + 12 prediction = 37 tests)
   - Preprocessing functions
   - Model architecture validation
   - Prediction pipeline components

2. **Integration Tests** (9 tests)
   - End-to-end flow
   - Character type variations
   - Edge cases
   - Performance benchmarks

3. **Test Automation**
   - `tests/run_tests.py` - Single command execution
   - Unittest framework
   - 100% pass rate requirement

### Performance Benchmarks

```python
def test_prediction_latency():
    """Measure warm prediction time"""
    # Warm up
    predict_character(model, test_image, labels)
    
    # Measure
    start = time.time()
    result = predict_character(model, test_image, labels)
    elapsed = time.time() - start
    
    assert elapsed < 0.1  # Must be under 100ms
```

---

## Deployment Configuration

### Streamlit Configuration

`.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"

[server]
headless = true
enableCORS = false
port = 8501

[browser]
gatherUsageStats = false
```

### Requirements

`requirements.txt`:
```
tensorflow>=2.20.0
streamlit>=1.25.0
streamlit-drawable-canvas>=0.9.0
numpy>=1.23.0
opencv-python-headless>=4.7.0
matplotlib>=3.10.0
scikit-learn>=1.2.0
pandas>=1.5.0
emnist
```

---

## Key Technical Decisions

### 1. EMNIST ByClass vs MNIST
**Decision**: Use EMNIST ByClass  
**Rationale**: 
- 62 classes vs 10 (digits only)
- Real-world applicability for alphanumeric recognition
- Same 28×28 format for easy adaptation

### 2. CNN Architecture
**Decision**: 3 conv blocks + 2 dense layers  
**Rationale**:
- Proven architecture for image classification
- 1.7M parameters balances capacity and efficiency
- Batch normalization improves training stability

### 3. In-place Preprocessing
**Decision**: Modify arrays in-place during training  
**Rationale**:
- Avoid RAM overflow with 697K samples
- Faster processing without data copying
- Enables training on memory-constrained systems

### 4. Canvas RGB vs Alpha
**Decision**: Use RGB channels for canvas preprocessing  
**Rationale**:
- Streamlit canvas draws on RGB, not alpha
- Grayscale conversion from RGB more reliable
- Fixed "all black" preprocessing bug

### 5. Dynamic Canvas Key
**Decision**: Increment key on clear button  
**Rationale**:
- Forces Streamlit to create new canvas component
- Achieves true canvas reset
- Simple implementation with session state

---

## Known Limitations

1. **Single Character Only**: Cannot recognize multiple characters in one drawing
2. **Similar Characters**: May confuse visually similar characters (e.g., 'O' and '0', 'l' and '1')
3. **Drawing Style Sensitivity**: Works best with clear, centered characters
4. **Memory Constraints**: Full dataset training requires 8-10GB RAM

---

## Future Improvements

1. **Model Enhancements**:
   - Ensemble methods for higher accuracy
   - Character segmentation for multi-character recognition
   - Active learning from user feedback

2. **UI Improvements**:
   - Undo/redo functionality
   - Drawing tips and examples
   - Real-time prediction as user draws

3. **Deployment**:
   - Model versioning system
   - A/B testing framework
   - Usage analytics dashboard

---

**Last Updated**: December 10, 2025  
**Version**: 1.0.0
