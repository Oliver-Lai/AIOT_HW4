"""
Unit tests for preprocessing functions.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.preprocessing import (
    normalize_images,
    reshape_images,
    one_hot_encode_labels,
    create_train_val_split
)


class TestPreprocessingFunctions(unittest.TestCase):
    """Test suite for preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data with balanced classes for stratified split
        # Use 620 samples (10 per class for 62 classes)
        self.sample_images = np.random.randint(0, 256, (620, 28, 28), dtype=np.uint8)
        # Ensure each class has exactly 10 samples
        self.sample_labels = np.repeat(np.arange(62), 10).astype(np.int32)
    
    def test_normalize_images_range(self):
        """Test that normalized images are in [0, 1] range."""
        normalized = normalize_images(self.sample_images)
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
        self.assertEqual(normalized.dtype, np.float32)
    
    def test_normalize_images_shape(self):
        """Test that normalization preserves shape."""
        normalized = normalize_images(self.sample_images)
        self.assertEqual(normalized.shape, self.sample_images.shape)
    
    def test_normalize_images_values(self):
        """Test specific normalization values."""
        test_img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        normalized = normalize_images(test_img)
        np.testing.assert_array_almost_equal(
            normalized, 
            np.array([[[0.0, 128/255, 1.0]]], dtype=np.float32),
            decimal=5
        )
    
    def test_reshape_images_adds_channel(self):
        """Test that reshape adds channel dimension."""
        reshaped = reshape_images(self.sample_images)
        self.assertEqual(reshaped.shape, (620, 28, 28, 1))
    
    def test_reshape_images_preserves_data(self):
        """Test that reshape preserves pixel values."""
        reshaped = reshape_images(self.sample_images)
        np.testing.assert_array_equal(
            reshaped[:, :, :, 0], 
            self.sample_images
        )
    
    def test_one_hot_encode_shape(self):
        """Test one-hot encoding produces correct shape."""
        encoded = one_hot_encode_labels(self.sample_labels, num_classes=62)
        self.assertEqual(encoded.shape, (620, 62))
    
    def test_one_hot_encode_values(self):
        """Test one-hot encoding produces correct values."""
        test_labels = np.array([0, 5, 61])
        encoded = one_hot_encode_labels(test_labels, num_classes=62)
        
        # Check that each row has exactly one 1
        self.assertTrue(np.all(encoded.sum(axis=1) == 1))
        
        # Check specific encodings
        self.assertEqual(encoded[0, 0], 1)
        self.assertEqual(encoded[1, 5], 1)
        self.assertEqual(encoded[2, 61], 1)
    
    def test_train_val_split_sizes(self):
        """Test that split produces correct sizes."""
        x_train, x_val, y_train, y_val = create_train_val_split(
            self.sample_images, 
            self.sample_labels, 
            val_size=0.2
        )
        
        self.assertEqual(len(x_train), 496)  # 80% of 620
        self.assertEqual(len(x_val), 124)    # 20% of 620
        self.assertEqual(len(y_train), 496)
        self.assertEqual(len(y_val), 124)
    
    def test_train_val_split_no_overlap(self):
        """Test that train and val sets don't overlap."""
        x_train, x_val, y_train, y_val = create_train_val_split(
            self.sample_images, 
            self.sample_labels, 
            val_size=0.2,
            random_state=42
        )
        
        # Check total size
        self.assertEqual(len(x_train) + len(x_val), len(self.sample_images))
    
    def test_train_val_split_reproducible(self):
        """Test that split is reproducible with same random_state."""
        split1 = create_train_val_split(
            self.sample_images, 
            self.sample_labels, 
            val_size=0.2,
            random_state=42
        )
        
        split2 = create_train_val_split(
            self.sample_images, 
            self.sample_labels, 
            val_size=0.2,
            random_state=42
        )
        
        np.testing.assert_array_equal(split1[0], split2[0])  # x_train
        np.testing.assert_array_equal(split1[2], split2[2])  # y_train
    
    def test_edge_case_empty_array(self):
        """Test handling of empty arrays."""
        empty_images = np.array([]).reshape(0, 28, 28).astype(np.uint8)
        normalized = normalize_images(empty_images)
        self.assertEqual(normalized.shape[0], 0)
    
    def test_edge_case_single_image(self):
        """Test handling of single image."""
        single_image = self.sample_images[0:1]
        normalized = normalize_images(single_image)
        self.assertEqual(normalized.shape, (1, 28, 28))


if __name__ == '__main__':
    unittest.main()
