"""
Unit tests for model architecture.
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tensorflow as tf
from models.cnn import create_cnn_model, compile_model, count_parameters


class TestCNNModel(unittest.TestCase):
    """Test suite for CNN model architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_shape = (28, 28, 1)
        self.num_classes = 62
        self.model = create_cnn_model(self.input_shape, self.num_classes)
    
    def test_model_creation(self):
        """Test that model is created successfully."""
        self.assertIsNotNone(self.model)
        self.assertIsInstance(self.model, tf.keras.Model)
    
    def test_input_shape(self):
        """Test that model accepts correct input shape."""
        self.assertEqual(self.model.input_shape, (None, 28, 28, 1))
    
    def test_output_shape(self):
        """Test that model produces correct output shape."""
        self.assertEqual(self.model.output_shape, (None, 62))
    
    def test_output_activation(self):
        """Test that output layer uses softmax activation."""
        output_layer = self.model.layers[-1]
        self.assertEqual(output_layer.activation.__name__, 'softmax')
    
    def test_parameter_count(self):
        """Test that model has approximately 1-3M parameters."""
        trainable, non_trainable = count_parameters(self.model)
        total_params = trainable + non_trainable
        self.assertGreater(total_params, 1_000_000)
        self.assertLess(total_params, 3_000_000)
        print(f"\nModel has {total_params:,} parameters (trainable: {trainable:,}, non-trainable: {non_trainable:,})")
    
    def test_model_prediction_shape(self):
        """Test that model prediction produces correct shape."""
        dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
        prediction = self.model.predict(dummy_input, verbose=0)
        self.assertEqual(prediction.shape, (1, 62))
    
    def test_model_prediction_probabilities(self):
        """Test that predictions sum to 1 (proper probabilities)."""
        dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
        prediction = self.model.predict(dummy_input, verbose=0)
        
        # Probabilities should sum to ~1
        prob_sum = prediction.sum()
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
        
        # All probabilities should be between 0 and 1
        self.assertTrue(np.all(prediction >= 0))
        self.assertTrue(np.all(prediction <= 1))
    
    def test_model_batch_prediction(self):
        """Test that model handles batch predictions."""
        batch_size = 32
        dummy_input = np.random.rand(batch_size, 28, 28, 1).astype(np.float32)
        prediction = self.model.predict(dummy_input, verbose=0)
        self.assertEqual(prediction.shape, (batch_size, 62))
    
    def test_model_compilation(self):
        """Test that model compiles successfully."""
        try:
            compile_model(self.model, learning_rate=0.001)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Model compilation failed: {e}")
    
    def test_model_has_conv_layers(self):
        """Test that model contains convolutional layers."""
        conv_layers = [layer for layer in self.model.layers 
                      if isinstance(layer, tf.keras.layers.Conv2D)]
        self.assertGreater(len(conv_layers), 0)
        print(f"\nModel has {len(conv_layers)} convolutional layers")
    
    def test_model_has_dense_layers(self):
        """Test that model contains dense layers."""
        dense_layers = [layer for layer in self.model.layers 
                       if isinstance(layer, tf.keras.layers.Dense)]
        self.assertGreater(len(dense_layers), 0)
        print(f"Model has {len(dense_layers)} dense layers")
    
    def test_model_has_dropout(self):
        """Test that model contains dropout layers for regularization."""
        dropout_layers = [layer for layer in self.model.layers 
                         if isinstance(layer, tf.keras.layers.Dropout)]
        self.assertGreater(len(dropout_layers), 0)
        print(f"Model has {len(dropout_layers)} dropout layers")
    
    def test_different_input_sizes(self):
        """Test model with different batch sizes."""
        for batch_size in [1, 16, 64]:
            dummy_input = np.random.rand(batch_size, 28, 28, 1).astype(np.float32)
            prediction = self.model.predict(dummy_input, verbose=0)
            self.assertEqual(prediction.shape[0], batch_size)


if __name__ == '__main__':
    unittest.main()
