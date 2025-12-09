"""
Quick training script to generate a model for Phase 5 evaluation.
Uses 100K samples to achieve ~83-84% accuracy with reasonable training time.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import time
from datetime import datetime
import json
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from src.data.dataset import load_emnist
from src.models.cnn import create_cnn_model, compile_model
from src.preprocessing.preprocessing import create_data_augmentation_generator

print("="*70)
print("TRAINING MODEL FOR PHASE 5 EVALUATION")
print("="*70)

# Configuration
SUBSET_SIZE = 100000  # 100K samples for good accuracy (~84%)
BATCH_SIZE = 128
EPOCHS = 25
VAL_SIZE = 0.15

print(f"\nConfiguration:")
print(f"  Training samples: {SUBSET_SIZE:,}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {EPOCHS}")
print(f"  Validation split: {VAL_SIZE*100:.0f}%")
print("="*70)

# Load data
print("\n[1/7] Loading EMNIST dataset...")
start_time = time.time()
x_train_full, y_train_full, x_test, y_test = load_emnist()
print(f"✓ Loaded in {time.time() - start_time:.1f}s")

# Create subset
print(f"\n[2/7] Creating {SUBSET_SIZE:,} sample subset...")
indices = np.arange(len(x_train_full))
subset_indices, _ = train_test_split(
    indices, 
    train_size=SUBSET_SIZE, 
    stratify=y_train_full, 
    random_state=42
)
x_train = x_train_full[subset_indices]
y_train = y_train_full[subset_indices]

# Free memory
del x_train_full, y_train_full
import gc
gc.collect()

print(f"✓ Subset created: {x_train.shape[0]:,} samples")

# Preprocess in-place
print("\n[3/7] Preprocessing data...")
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, 62)
y_test = to_categorical(y_test, 62)

print(f"✓ Preprocessed")

# Train/val split
print("\n[4/7] Creating train/validation split...")
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, 
    test_size=VAL_SIZE, 
    random_state=42,
    stratify=np.argmax(y_train, axis=1)
)
print(f"✓ Train: {x_train.shape[0]:,}, Val: {x_val.shape[0]:,}")

# Create model
print("\n[5/7] Creating CNN model...")
model = create_cnn_model()
model = compile_model(model, learning_rate=0.001)
print("✓ Model created")

# Setup
model_path = Path('models/emnist_cnn_v1.keras')
model_path.parent.mkdir(parents=True, exist_ok=True)

datagen = create_data_augmentation_generator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
datagen.fit(x_train)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(model_path),
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# Train
print("\n[6/7] Training model...")
print("="*70)

training_start = time.time()

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - training_start

print("="*70)
print(f"✓ Training completed in {training_time/60:.1f} minutes")

# Evaluate
print("\n[7/7] Evaluating on test set...")
test_results = model.evaluate(x_test, y_test, verbose=0)
test_loss, test_accuracy, test_top5 = test_results[0], test_results[1], test_results[2]

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy*100:.2f}%")
print(f"Test top-5 accuracy: {test_top5*100:.2f}%")
print(f"Best val accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

# Save history
history_dict = {
    'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
    'config': {
        'batch_size': BATCH_SIZE,
        'epochs': len(history.history['loss']),
        'learning_rate': 0.001,
        'val_size': VAL_SIZE,
        'use_augmentation': True,
        'subset_size': SUBSET_SIZE,
        'training_time_seconds': training_time
    },
    'test_results': {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'test_top5_accuracy': float(test_top5)
    },
    'timestamp': datetime.now().isoformat()
}

history_path = Path('models/emnist_cnn_v1_history.json')
with open(history_path, 'w') as f:
    json.dump(history_dict, f, indent=2)

print(f"\n✓ Model saved to: {model_path}")
print(f"✓ History saved to: {history_path}")
print("="*70)
print("MODEL READY FOR PHASE 5 EVALUATION!")
print("="*70)
