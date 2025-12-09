"""
Quick test script for memory-efficient training.
Tests with a small subset to verify everything works.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.data.dataset import load_emnist
from src.models.cnn import create_cnn_model, compile_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

print("="*70)
print("MEMORY-EFFICIENT TRAINING TEST")
print("="*70)

# Load data
print("\n[1/5] Loading dataset...")
x_train_full, y_train_full, x_test, y_test = load_emnist()
print(f"✓ Full dataset: {x_train_full.shape[0]:,} training samples")

# Use small subset for testing
TEST_SIZE = 10000
print(f"\n[2/5] Creating test subset ({TEST_SIZE:,} samples)...")
indices = np.arange(len(x_train_full))
subset_indices, _ = train_test_split(
    indices, 
    train_size=TEST_SIZE, 
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
print("\n[3/5] Preprocessing (in-place)...")
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 62)
y_test = to_categorical(y_test, 62)
print(f"✓ Preprocessed: {x_train.shape}")

# Split
print("\n[4/5] Creating train/val split...")
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.15, random_state=42,
    stratify=np.argmax(y_train, axis=1)
)
print(f"✓ Train: {x_train.shape[0]:,}, Val: {x_val.shape[0]:,}")

# Create model
print("\n[5/5] Creating model...")
model = create_cnn_model()
model = compile_model(model)
print("✓ Model created")

# Quick training test (1 epoch)
print("\nRunning 1 epoch test...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=1,
    validation_data=(x_val, y_val),
    verbose=1
)

print("\n" + "="*70)
print("TEST SUCCESSFUL! ✓")
print("="*70)
print(f"Training accuracy: {history.history['accuracy'][0]*100:.2f}%")
print(f"Validation accuracy: {history.history['val_accuracy'][0]*100:.2f}%")
print("\nMemory-efficient approach is working correctly.")
print("You can now train with larger subsets or full dataset.")
print("="*70)
