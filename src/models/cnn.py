"""
CNN model architecture for EMNIST character recognition.

This module provides the CNN model definition with:
- 3 convolutional blocks with batch normalization
- MaxPooling layers for dimensionality reduction
- Dense layers with dropout for regularization
- Softmax output for 62-class classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Tuple, Optional


def create_cnn_model(
    input_shape: Tuple[int, int, int] = (28, 28, 1),
    num_classes: int = 62,
    conv_filters: Tuple[int, int, int] = (64, 128, 256),
    dense_units: Tuple[int, int] = (512, 256),
    dropout_rates: Tuple[float, float] = (0.5, 0.3),
    kernel_size: int = 3
) -> keras.Model:
    """
    Create a CNN model for EMNIST character recognition.
    
    Architecture:
    - Conv Block 1: 64 filters -> BatchNorm -> ReLU -> MaxPool
    - Conv Block 2: 128 filters -> BatchNorm -> ReLU -> MaxPool
    - Conv Block 3: 256 filters -> BatchNorm -> ReLU -> MaxPool
    - Flatten
    - Dense 512 -> Dropout 0.5
    - Dense 256 -> Dropout 0.3
    - Dense 62 (Softmax)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes (default: 62 for EMNIST ByClass)
        conv_filters: Number of filters in each conv block
        dense_units: Number of units in dense layers
        dropout_rates: Dropout rates for dense layers
        kernel_size: Size of convolutional kernels
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential(name='EMNIST_CNN')
    
    # Convolutional Block 1
    model.add(layers.Conv2D(
        conv_filters[0], 
        kernel_size, 
        padding='same',
        input_shape=input_shape,
        name='conv1'
    ))
    model.add(layers.BatchNormalization(name='bn1'))
    model.add(layers.Activation('relu', name='relu1'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool1'))
    
    # Convolutional Block 2
    model.add(layers.Conv2D(
        conv_filters[1], 
        kernel_size, 
        padding='same',
        name='conv2'
    ))
    model.add(layers.BatchNormalization(name='bn2'))
    model.add(layers.Activation('relu', name='relu2'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool2'))
    
    # Convolutional Block 3
    model.add(layers.Conv2D(
        conv_filters[2], 
        kernel_size, 
        padding='same',
        name='conv3'
    ))
    model.add(layers.BatchNormalization(name='bn3'))
    model.add(layers.Activation('relu', name='relu3'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name='pool3'))
    
    # Flatten and Dense Layers
    model.add(layers.Flatten(name='flatten'))
    
    # Dense Layer 1
    model.add(layers.Dense(dense_units[0], name='dense1'))
    model.add(layers.BatchNormalization(name='bn_dense1'))
    model.add(layers.Activation('relu', name='relu_dense1'))
    model.add(layers.Dropout(dropout_rates[0], name='dropout1'))
    
    # Dense Layer 2
    model.add(layers.Dense(dense_units[1], name='dense2'))
    model.add(layers.BatchNormalization(name='bn_dense2'))
    model.add(layers.Activation('relu', name='relu_dense2'))
    model.add(layers.Dropout(dropout_rates[1], name='dropout2'))
    
    # Output Layer
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    metrics: Optional[list] = None
) -> keras.Model:
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
        metrics: List of metrics to track (default: accuracy and top-5 accuracy)
        
    Returns:
        Compiled Keras model
    """
    if metrics is None:
        metrics = [
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=metrics
    )
    
    return model


def get_model_summary(model: keras.Model) -> str:
    """
    Get model summary as a string.
    
    Args:
        model: Keras model
        
    Returns:
        String representation of model summary
    """
    from io import StringIO
    import sys
    
    # Redirect stdout to capture summary
    old_stdout = sys.stdout
    sys.stdout = summary_buffer = StringIO()
    
    model.summary()
    
    # Restore stdout
    sys.stdout = old_stdout
    
    return summary_buffer.getvalue()


def count_parameters(model: keras.Model) -> Tuple[int, int]:
    """
    Count trainable and non-trainable parameters in the model.
    
    Args:
        model: Keras model
        
    Returns:
        Tuple of (trainable_params, non_trainable_params)
    """
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    return trainable, non_trainable


if __name__ == "__main__":
    """Test model creation and verify architecture."""
    print("Creating EMNIST CNN model...")
    
    # Create model
    model = create_cnn_model()
    
    # Compile model
    model = compile_model(model)
    
    # Print model summary
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE")
    print("="*70)
    model.summary()
    
    # Count parameters
    trainable, non_trainable = count_parameters(model)
    total = trainable + non_trainable
    
    print("\n" + "="*70)
    print("PARAMETER COUNT")
    print("="*70)
    print(f"Trainable parameters:     {trainable:,}")
    print(f"Non-trainable parameters: {non_trainable:,}")
    print(f"Total parameters:         {total:,}")
    
    # Verify input/output shapes
    print("\n" + "="*70)
    print("INPUT/OUTPUT VERIFICATION")
    print("="*70)
    
    import numpy as np
    
    # Test with sample input
    sample_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
    sample_output = model.predict(sample_input, verbose=0)
    
    print(f"Input shape:  {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")
    print(f"Output sum:   {sample_output.sum():.4f} (should be ~1.0)")
    print(f"Max prob:     {sample_output.max():.4f}")
    print(f"Min prob:     {sample_output.min():.6f}")
    
    # Verify architecture constraints
    print("\n" + "="*70)
    print("ARCHITECTURE VALIDATION")
    print("="*70)
    
    checks = []
    
    # Check parameter count (should be 1-3M)
    param_check = 1_000_000 <= total <= 3_000_000
    checks.append(("Parameter count in [1M, 3M]", param_check))
    
    # Check input shape
    input_check = model.input_shape == (None, 28, 28, 1)
    checks.append(("Input shape is (None, 28, 28, 1)", input_check))
    
    # Check output shape
    output_check = model.output_shape == (None, 62)
    checks.append(("Output shape is (None, 62)", output_check))
    
    # Check output is probability distribution
    prob_check = abs(sample_output.sum() - 1.0) < 0.001
    checks.append(("Output sums to 1.0 (softmax)", prob_check))
    
    # Print results
    all_passed = True
    for check_name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("All validation checks passed! ✓")
    else:
        print("Some validation checks failed! ✗")
    print("="*70)
