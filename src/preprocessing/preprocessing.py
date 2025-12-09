"""
Data preprocessing utilities for EMNIST dataset.

This module provides functions for:
- Normalization (scaling pixel values)
- Reshaping (adding channel dimensions)
- One-hot encoding (converting labels)
- Train/validation splitting
- Data augmentation
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def normalize_images(images: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values from [0, 255] to [0, 1].
    
    Args:
        images: Array of images with shape (n_samples, height, width)
        
    Returns:
        Normalized images with values in [0, 1]
    """
    if images.dtype != np.float32:
        images = images.astype(np.float32)
    return images / 255.0


def reshape_images(images: np.ndarray) -> np.ndarray:
    """
    Add channel dimension to images for CNN input.
    
    Args:
        images: Array of images with shape (n_samples, height, width)
        
    Returns:
        Reshaped images with shape (n_samples, height, width, 1)
    """
    if len(images.shape) == 3:
        return images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    return images


def one_hot_encode_labels(labels: np.ndarray, num_classes: int = 62) -> np.ndarray:
    """
    Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: Array of integer labels with shape (n_samples,)
        num_classes: Number of classes (default: 62 for EMNIST ByClass)
        
    Returns:
        One-hot encoded labels with shape (n_samples, num_classes)
    """
    return to_categorical(labels, num_classes=num_classes)


def preprocess_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int = 62
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply full preprocessing pipeline to training and test data.
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        num_classes: Number of classes
        
    Returns:
        Tuple of (preprocessed x_train, y_train, x_test, y_test)
    """
    # Normalize
    x_train = normalize_images(x_train)
    x_test = normalize_images(x_test)
    
    # Reshape
    x_train = reshape_images(x_train)
    x_test = reshape_images(x_test)
    
    # One-hot encode labels
    y_train = one_hot_encode_labels(y_train, num_classes)
    y_test = one_hot_encode_labels(y_test, num_classes)
    
    return x_train, y_train, x_test, y_test


def create_train_val_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into training and validation sets.
    
    Args:
        x_train: Training images
        y_train: Training labels (can be one-hot encoded or integer)
        val_size: Fraction of data to use for validation (default: 0.15)
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class distribution in splits
        
    Returns:
        Tuple of (x_train_split, x_val, y_train_split, y_val)
    """
    # If labels are one-hot encoded, use argmax for stratification
    stratify_labels = None
    if stratify:
        if len(y_train.shape) > 1:
            stratify_labels = np.argmax(y_train, axis=1)
        else:
            stratify_labels = y_train
    
    x_train_split, x_val, y_train_split, y_val = train_test_split(
        x_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_labels
    )
    
    return x_train_split, x_val, y_train_split, y_val


def create_data_augmentation_generator(
    rotation_range: int = 15,
    width_shift_range: float = 0.1,
    height_shift_range: float = 0.1,
    zoom_range: float = 0.1,
    fill_mode: str = 'nearest'
) -> ImageDataGenerator:
    """
    Create an ImageDataGenerator for data augmentation.
    
    Args:
        rotation_range: Degree range for random rotations (default: ±15°)
        width_shift_range: Fraction of width for horizontal shifts (default: ±10%)
        height_shift_range: Fraction of height for vertical shifts (default: ±10%)
        zoom_range: Range for random zoom (default: ±10%)
        fill_mode: Points outside boundaries filled according to this mode
        
    Returns:
        Configured ImageDataGenerator instance
    """
    return ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        zoom_range=zoom_range,
        fill_mode=fill_mode
    )


def visualize_augmentation(
    image: np.ndarray,
    datagen: ImageDataGenerator,
    num_samples: int = 9
) -> np.ndarray:
    """
    Generate augmented samples from a single image for visualization.
    
    Args:
        image: Single image with shape (height, width, channels)
        datagen: Configured ImageDataGenerator
        num_samples: Number of augmented samples to generate
        
    Returns:
        Array of augmented images with shape (num_samples, height, width, channels)
    """
    # Reshape image to (1, height, width, channels) for datagen
    if len(image.shape) == 2:
        image = image.reshape(1, image.shape[0], image.shape[1], 1)
    elif len(image.shape) == 3:
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    
    augmented_images = []
    for batch in datagen.flow(image, batch_size=1):
        augmented_images.append(batch[0])
        if len(augmented_images) >= num_samples:
            break
    
    return np.array(augmented_images)


if __name__ == "__main__":
    """Test preprocessing functions with sample data."""
    print("Testing preprocessing functions...")
    
    # Create sample data (at least 10 samples per class for stratification)
    sample_images = np.random.randint(0, 256, size=(620, 28, 28), dtype=np.uint8)
    sample_labels = np.tile(np.arange(62), 10)  # 10 samples per class
    
    print(f"\nOriginal images shape: {sample_images.shape}")
    print(f"Original images dtype: {sample_images.dtype}")
    print(f"Original images range: [{sample_images.min()}, {sample_images.max()}]")
    
    # Test normalization
    normalized = normalize_images(sample_images)
    print(f"\nNormalized images dtype: {normalized.dtype}")
    print(f"Normalized images range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    assert normalized.min() >= 0.0 and normalized.max() <= 1.0, "Normalization failed"
    print("✓ Normalization test passed")
    
    # Test reshaping
    reshaped = reshape_images(normalized)
    print(f"\nReshaped images shape: {reshaped.shape}")
    assert reshaped.shape == (620, 28, 28, 1), "Reshaping failed"
    print("✓ Reshaping test passed")
    
    # Test one-hot encoding
    one_hot = one_hot_encode_labels(sample_labels, num_classes=62)
    print(f"\nOne-hot encoded labels shape: {one_hot.shape}")
    assert one_hot.shape == (620, 62), "One-hot encoding failed"
    assert np.allclose(one_hot.sum(axis=1), 1.0), "One-hot encoding sum != 1"
    print("✓ One-hot encoding test passed")
    
    # Test train/val split
    x_train, x_val, y_train, y_val = create_train_val_split(
        reshaped, one_hot, val_size=0.15, random_state=42
    )
    print(f"\nTrain/Val split results:")
    print(f"  Training set: {x_train.shape[0]} samples ({x_train.shape[0]/620*100:.1f}%)")
    print(f"  Validation set: {x_val.shape[0]} samples ({x_val.shape[0]/620*100:.1f}%)")
    assert abs(x_val.shape[0] / 620 - 0.15) < 0.01, "Val split size incorrect"
    print("✓ Train/Val split test passed")
    
    # Test data augmentation generator
    datagen = create_data_augmentation_generator()
    print(f"\nData augmentation generator created")
    print(f"  Rotation range: ±15°")
    print(f"  Shift range: ±10%")
    print(f"  Zoom range: ±10%")
    
    # Test augmentation visualization
    single_image = reshaped[0]
    augmented = visualize_augmentation(single_image, datagen, num_samples=9)
    print(f"\nAugmented images shape: {augmented.shape}")
    assert augmented.shape[0] == 9, "Augmentation generation failed"
    assert augmented.min() >= 0.0 and augmented.max() <= 1.0, "Augmented images out of range"
    print("✓ Augmentation visualization test passed")
    
    print("\n" + "="*50)
    print("All preprocessing tests passed! ✓")
    print("="*50)
