"""
Complete preprocessing pipeline test with EMNIST data.

This script demonstrates the full preprocessing workflow:
1. Load EMNIST dataset
2. Apply all preprocessing steps
3. Create train/val split
4. Verify data integrity
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import load_emnist
from src.preprocessing.preprocessing import (
    preprocess_data,
    create_train_val_split,
    create_data_augmentation_generator
)

def main():
    print("="*70)
    print("EMNIST PREPROCESSING PIPELINE TEST")
    print("="*70)
    
    # Step 1: Load data
    print("\n1. Loading EMNIST dataset...")
    x_train, y_train, x_test, y_test = load_emnist()
    print(f"   ✓ Training: {x_train.shape[0]:,} samples")
    print(f"   ✓ Test: {x_test.shape[0]:,} samples")
    
    # Step 2: Apply preprocessing
    print("\n2. Applying preprocessing (normalize, reshape, one-hot encode)...")
    x_train_prep, y_train_prep, x_test_prep, y_test_prep = preprocess_data(
        x_train, y_train, x_test, y_test, num_classes=62
    )
    print(f"   ✓ Training images: {x_train_prep.shape}")
    print(f"   ✓ Training labels: {y_train_prep.shape}")
    print(f"   ✓ Pixel range: [{x_train_prep.min():.4f}, {x_train_prep.max():.4f}]")
    
    # Step 3: Create train/val split
    print("\n3. Creating train/validation split (85/15)...")
    x_train_split, x_val, y_train_split, y_val = create_train_val_split(
        x_train_prep, y_train_prep, val_size=0.15, random_state=42, stratify=True
    )
    print(f"   ✓ Training: {x_train_split.shape[0]:,} samples ({x_train_split.shape[0]/x_train_prep.shape[0]*100:.1f}%)")
    print(f"   ✓ Validation: {x_val.shape[0]:,} samples ({x_val.shape[0]/x_train_prep.shape[0]*100:.1f}%)")
    
    # Step 4: Verify class distribution
    print("\n4. Verifying stratification...")
    train_classes = np.argmax(y_train_split, axis=1)
    val_classes = np.argmax(y_val, axis=1)
    
    train_dist = np.bincount(train_classes, minlength=62) / len(train_classes)
    val_dist = np.bincount(val_classes, minlength=62) / len(val_classes)
    correlation = np.corrcoef(train_dist, val_dist)[0, 1]
    
    print(f"   ✓ Class distribution correlation: {correlation:.6f}")
    if correlation > 0.99:
        print("   ✓ Distributions are well balanced")
    
    # Step 5: Configure augmentation
    print("\n5. Configuring data augmentation...")
    datagen = create_data_augmentation_generator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    print("   ✓ Augmentation generator ready")
    print("      - Rotation: ±15°")
    print("      - Width/Height shift: ±10%")
    print("      - Zoom: ±10%")
    
    # Step 6: Test augmentation
    print("\n6. Testing augmentation on sample batch...")
    test_batch = x_train_split[:32]
    aug_count = 0
    for aug_batch in datagen.flow(test_batch, batch_size=32):
        aug_count += 1
        if aug_count >= 1:
            break
    
    print(f"   ✓ Generated augmented batch: {aug_batch.shape}")
    print(f"   ✓ Augmented pixel range: [{aug_batch.min():.4f}, {aug_batch.max():.4f}]")
    print(f"   ✓ No NaN values: {not np.isnan(aug_batch).any()}")
    print(f"   ✓ No Inf values: {not np.isinf(aug_batch).any()}")
    
    # Final summary
    print("\n" + "="*70)
    print("PREPROCESSING PIPELINE TEST COMPLETE ✓")
    print("="*70)
    print(f"\n📊 Final Dataset Configuration:")
    print(f"   Training set:   {x_train_split.shape[0]:>8,} samples → {x_train_split.shape}")
    print(f"   Validation set: {x_val.shape[0]:>8,} samples → {x_val.shape}")
    print(f"   Test set:       {x_test_prep.shape[0]:>8,} samples → {x_test_prep.shape}")
    print(f"\n✅ All data is preprocessed and ready for model training!")
    print("="*70)

if __name__ == "__main__":
    main()
