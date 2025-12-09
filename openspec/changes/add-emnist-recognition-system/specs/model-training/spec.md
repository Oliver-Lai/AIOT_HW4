# Specification: Model Training

## ADDED Requirements

### Requirement: EMNIST Dataset Loading Must Download ByClass Dataset
The system SHALL download and load the EMNIST ByClass dataset for training character recognition models.

**Properties**:
- Dataset source: EMNIST ByClass (62 classes: 10 digits + 26 uppercase + 26 lowercase letters)
- Image format: 28x28 grayscale images
- Data splits: Training set (697,932 images), Test set (116,323 images)
- Label encoding: Integer labels 0-61 mapped to characters

#### Scenario: Developer downloads EMNIST dataset
**Given** a Python environment with dataset libraries installed  
**When** the developer runs the dataset loading script  
**Then** the EMNIST ByClass dataset is downloaded to the `data/emnist/` directory  
**And** both training and test sets are accessible as NumPy arrays  
**And** a label mapping file is created mapping indices to characters

#### Scenario: System loads dataset for training
**Given** the EMNIST dataset exists in local storage  
**When** the training script initializes  
**Then** training images are loaded with shape (N, 28, 28)  
**And** training labels are loaded with shape (N,) containing integer class indices  
**And** test images and labels are loaded separately  
**And** data loading completes within 30 seconds

### Requirement: Data Preprocessing Pipeline Must Normalize and Augment Data
Training data SHALL be preprocessed and augmented to improve model generalization.

**Properties**:
- Normalization: Pixel values scaled from [0, 255] to [0, 1]
- Reshaping: Images reshaped to (28, 28, 1) for CNN input
- Label encoding: Integer labels converted to one-hot encoded vectors
- Train/validation split: 85/15 split from training data
- Data augmentation: Random rotation (±15°), width/height shift (±10%)

#### Scenario: Preprocess training images
**Given** raw EMNIST training images with pixel values 0-255  
**When** the preprocessing pipeline runs  
**Then** pixel values are normalized to range [0, 1]  
**And** images are reshaped to (N, 28, 28, 1) format  
**And** labels are one-hot encoded to (N, 62) format

#### Scenario: Apply data augmentation
**Given** preprocessed training images  
**When** data augmentation is applied  
**Then** random rotations up to ±15 degrees are applied to 50% of images  
**And** random width and height shifts up to 10% are applied  
**And** augmented images remain within valid pixel range [0, 1]  
**And** original image count increases by 2x through augmentation

#### Scenario: Split training data for validation
**Given** the full EMNIST training set  
**When** the train/validation split is performed  
**Then** 85% of data is allocated to training  
**And** 15% of data is allocated to validation  
**And** class distribution is balanced across both splits  
**And** no data leakage occurs between training and validation sets

### Requirement: CNN Model Architecture Must Support Character Classification
A convolutional neural network SHALL be defined with appropriate layers for character classification.

**Properties**:
- Input shape: (28, 28, 1) grayscale images
- Convolutional layers: 3 blocks with increasing filters (32, 64, 128)
- Pooling: MaxPooling2D with 2x2 pool size after each conv block
- Normalization: Batch normalization after each convolutional layer
- Dense layers: 2 fully connected layers (256, 128 units) with ReLU activation
- Dropout: 0.5 after first dense layer, 0.3 after second dense layer
- Output layer: 62 units with softmax activation
- Total parameters: Approximately 1-3 million trainable parameters

#### Scenario: Initialize CNN model
**Given** the model architecture definition  
**When** the model is instantiated  
**Then** the model input shape is (None, 28, 28, 1)  
**And** the model output shape is (None, 62)  
**And** all layers are properly connected  
**And** model summary shows layer dimensions and parameter counts

#### Scenario: Model accepts batch input
**Given** a trained CNN model  
**When** a batch of 32 images is fed to the model  
**Then** the model processes all images in a single forward pass  
**And** output predictions have shape (32, 62)  
**And** inference completes in less than 100ms on CPU

### Requirement: Model Training Process Must Use Proper Optimization
The CNN model SHALL be trained on EMNIST data with proper optimization and monitoring.

**Properties**:
- Optimizer: Adam with learning rate 0.001
- Loss function: Categorical crossentropy
- Metrics: Accuracy, top-5 accuracy
- Batch size: 128
- Epochs: Maximum 50 with early stopping
- Early stopping: Patience of 5 epochs monitoring validation loss
- Callbacks: ModelCheckpoint saves best model based on validation accuracy

#### Scenario: Train model on EMNIST dataset
**Given** preprocessed training and validation data  
**And** an initialized CNN model  
**When** training starts  
**Then** the model trains for up to 50 epochs  
**And** training loss decreases over epochs  
**And** validation accuracy is evaluated after each epoch  
**And** training stops early if validation loss doesn't improve for 5 consecutive epochs

#### Scenario: Monitor training progress
**Given** an ongoing training session  
**When** each epoch completes  
**Then** training loss and accuracy are logged  
**And** validation loss and accuracy are logged  
**And** current epoch number and time are displayed  
**And** progress is visualized in training logs

#### Scenario: Save best model checkpoint
**Given** training is in progress  
**When** validation accuracy improves  
**Then** the current model weights are saved to `models/emnist_cnn_best.h5`  
**And** the saved model includes full architecture and weights  
**And** previous best model is overwritten  
**And** a label mapping JSON file is saved alongside the model

### Requirement: Model Evaluation Must Achieve Target Accuracy
The trained model SHALL be evaluated on the test set with comprehensive metrics.

**Properties**:
- Test accuracy target: ≥85%
- Top-5 accuracy target: ≥95%
- Evaluation metrics: Accuracy, precision, recall, F1-score per class
- Confusion matrix: 62x62 matrix showing prediction patterns
- Inference time: <50ms per image on CPU

#### Scenario: Evaluate model on test set
**Given** a trained model saved to disk  
**And** the EMNIST test dataset  
**When** evaluation is performed  
**Then** overall test accuracy is calculated and displayed  
**And** top-5 test accuracy is calculated and displayed  
**And** per-class metrics (precision, recall, F1) are computed  
**And** results are saved to `models/evaluation_results.json`

#### Scenario: Generate confusion matrix
**Given** model predictions on the test set  
**When** the confusion matrix is generated  
**Then** a 62x62 matrix is created  
**And** rows represent true labels, columns represent predictions  
**And** matrix values show prediction counts for each class pair  
**And** the matrix is saved as a PNG visualization

#### Scenario: Measure inference performance
**Given** a trained model loaded in memory  
**When** 1000 test images are predicted individually  
**Then** average inference time per image is calculated  
**And** inference time is under 50ms per image on CPU  
**And** inference time is under 10ms per image on GPU if available

### Requirement: Model Export Must Be Deployment-Ready
The trained model SHALL be exported in a format suitable for deployment.

**Properties**:
- Model format: Keras HDF5 (.h5) or SavedModel format
- Model size: <100MB for Streamlit Cloud compatibility
- Auxiliary files: label_mapping.json with index-to-character mappings
- Version tracking: Models named with version numbers (e.g., emnist_cnn_v1.h5)

#### Scenario: Export trained model
**Given** a successfully trained CNN model  
**When** the export process runs  
**Then** the model is saved to `models/emnist_cnn_v1.h5`  
**And** the file size is verified to be under 100MB  
**And** model can be loaded back using `tf.keras.models.load_model()`  
**And** loaded model produces identical predictions to original

#### Scenario: Save label mapping
**Given** the EMNIST class indices and corresponding characters  
**When** label mapping is exported  
**Then** a JSON file is created at `models/label_mapping.json`  
**And** the JSON contains all 62 class mappings  
**And** mappings follow format: {"0": "0", "1": "1", ..., "35": "Z", ...}  
**And** the file is valid JSON and can be parsed without errors
