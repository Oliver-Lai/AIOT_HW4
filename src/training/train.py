"""
Training script for EMNIST CNN model.

This script handles:
- Data loading and preprocessing
- Model creation and compilation
- Training with callbacks (early stopping, checkpointing)
- Training history logging
- Model saving
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.dataset import load_emnist
from src.preprocessing.preprocessing import (
    preprocess_data,
    create_train_val_split,
    create_data_augmentation_generator
)
from src.models.cnn import create_cnn_model, compile_model


def setup_callbacks(
    model_save_path: Path,
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 3
) -> list:
    """
    Set up training callbacks.
    
    Args:
        model_save_path: Path to save best model
        early_stopping_patience: Number of epochs with no improvement before stopping
        reduce_lr_patience: Number of epochs with no improvement before reducing LR
        
    Returns:
        List of Keras callbacks
    """
    callbacks = []
    
    # Model checkpoint - save best model
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=str(model_save_path),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping_callback)
    
    # Reduce learning rate on plateau
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=1e-6,
        verbose=1
    )
    callbacks.append(reduce_lr_callback)
    
    return callbacks


def train_model(
    batch_size: int = 128,
    epochs: int = 50,
    learning_rate: float = 0.001,
    val_size: float = 0.15,
    use_augmentation: bool = True,
    model_name: str = "emnist_cnn_v1",
    save_dir: str = "models"
):
    """
    Train the EMNIST CNN model.
    
    Args:
        batch_size: Batch size for training
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate
        val_size: Fraction of training data for validation
        use_augmentation: Whether to use data augmentation
        model_name: Name for saving the model
        save_dir: Directory to save model and history
    """
    print("="*70)
    print("EMNIST CNN TRAINING")
    print("="*70)
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Validation split: {val_size*100:.0f}%")
    print(f"  Data augmentation: {use_augmentation}")
    print(f"  Model name: {model_name}")
    print("="*70)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n[1/6] Loading EMNIST dataset...")
    start_time = time.time()
    x_train, y_train, x_test, y_test = load_emnist()
    print(f"✓ Loaded in {time.time() - start_time:.1f}s")
    print(f"  Training: {x_train.shape[0]:,} samples")
    print(f"  Test: {x_test.shape[0]:,} samples")
    
    # Memory-efficient preprocessing: process in-place
    print("\n[2/6] Preprocessing data (memory-efficient)...")
    start_time = time.time()
    
    # Normalize in-place to save memory
    print("  Normalizing images...")
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # Reshape in-place
    print("  Reshaping images...")
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # One-hot encode labels
    print("  One-hot encoding labels...")
    from tensorflow.keras.utils import to_categorical
    y_train = to_categorical(y_train, num_classes=62)
    y_test = to_categorical(y_test, num_classes=62)
    
    print(f"✓ Preprocessed in {time.time() - start_time:.1f}s")
    print(f"  Training shape: {x_train.shape}")
    print(f"  Test shape: {x_test.shape}")
    
    # Create train/val split
    print("\n[3/6] Creating train/validation split...")
    start_time = time.time()
    x_train, x_val, y_train, y_val = create_train_val_split(
        x_train, y_train, val_size=val_size, random_state=42, stratify=True
    )
    print(f"✓ Split in {time.time() - start_time:.1f}s")
    print(f"  Training: {x_train.shape[0]:,} samples ({(1-val_size)*100:.0f}%)")
    print(f"  Validation: {x_val.shape[0]:,} samples ({val_size*100:.0f}%)")
    
    # Create model
    print("\n[4/6] Creating CNN model...")
    start_time = time.time()
    model = create_cnn_model()
    model = compile_model(model, learning_rate=learning_rate)
    print(f"✓ Model created in {time.time() - start_time:.1f}s")
    
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"  Total parameters: {trainable:,}")
    
    # Setup callbacks
    model_path = save_path / f"{model_name}.keras"
    callbacks = setup_callbacks(
        model_save_path=model_path,
        early_stopping_patience=5,
        reduce_lr_patience=3
    )
    
    # Prepare data augmentation
    if use_augmentation:
        print("\n[5/6] Setting up data augmentation...")
        datagen = create_data_augmentation_generator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        datagen.fit(x_train)
        print("✓ Data augmentation configured")
        print("  Rotation: ±15°")
        print("  Shifts: ±10%")
        print("  Zoom: ±10%")
    
    # Train model
    print(f"\n[6/6] Training model...")
    print("="*70)
    
    training_start = time.time()
    
    if use_augmentation:
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(x_train) // batch_size,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    training_time = time.time() - training_start
    
    print("="*70)
    print(f"✓ Training completed in {training_time/60:.1f} minutes")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(x_test, y_test, verbose=0)
    test_loss = test_results[0]
    test_accuracy = test_results[1]
    test_top5_accuracy = test_results[2]
    
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test accuracy: {test_accuracy*100:.2f}%")
    print(f"  Test top-5 accuracy: {test_top5_accuracy*100:.2f}%")
    
    # Save training history
    history_path = save_path / f"{model_name}_history.json"
    history_dict = {
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'config': {
            'batch_size': batch_size,
            'epochs': len(history.history['loss']),
            'learning_rate': learning_rate,
            'val_size': val_size,
            'use_augmentation': use_augmentation,
            'training_time_seconds': training_time
        },
        'test_results': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_accuracy),
            'test_top5_accuracy': float(test_top5_accuracy)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"\n✓ Training history saved to {history_path}")
    print(f"✓ Best model saved to {model_path}")
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"Final test accuracy: {test_accuracy*100:.2f}%")
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Epochs completed: {len(history.history['loss'])}")
    
    target_met = test_accuracy >= 0.85
    print(f"\nTarget (≥85% test accuracy): {'✓ MET' if target_met else '✗ NOT MET'}")
    print("="*70)
    
    return history, model


def main():
    """Main training function with command-line arguments."""
    parser = argparse.ArgumentParser(description='Train EMNIST CNN model')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation split size (default: 0.15)')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--model-name', type=str, default='emnist_cnn_v1',
                        help='Model name for saving (default: emnist_cnn_v1)')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='Directory to save model (default: models)')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        val_size=args.val_size,
        use_augmentation=not args.no_augmentation,
        model_name=args.model_name,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
