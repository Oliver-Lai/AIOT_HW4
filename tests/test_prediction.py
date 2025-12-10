"""
Unit tests for prediction pipeline (app.py functions).
"""

import unittest
import numpy as np
import cv2
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestPredictionPipeline(unittest.TestCase):
    """Test suite for prediction pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.canvas_size = 280
        self.model_path = "models/emnist_cnn_v1.keras"
        self.label_path = "models/label_mapping.json"
    
    def create_test_canvas(self, draw_something=True):
        """Create a test canvas with optional drawing."""
        # White background (RGBA)
        canvas = np.ones((self.canvas_size, self.canvas_size, 4), dtype=np.uint8) * 255
        
        if draw_something:
            # Draw a simple shape in black
            canvas[100:180, 100:180, 0:3] = 0
        
        return canvas
    
    def preprocess_canvas_image(self, canvas_data):
        """Preprocess canvas image (same as app.py)."""
        if canvas_data is None:
            return None
        
        # Convert RGBA to grayscale
        img_rgb = canvas_data[:, :, :3]
        img_gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Check if canvas is empty
        if img_gray.min() > 250:
            return None
        
        # Resize to 28x28
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Invert colors
        img_inverted = 255 - img_resized
        
        # Normalize to [0, 1]
        img_normalized = img_inverted.astype(np.float32) / 255.0
        
        # Reshape to (1, 28, 28, 1)
        img_final = img_normalized.reshape(1, 28, 28, 1)
        
        return img_final
    
    def test_preprocess_canvas_with_drawing(self):
        """Test preprocessing canvas with drawing."""
        canvas = self.create_test_canvas(draw_something=True)
        processed = self.preprocess_canvas_image(canvas)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (1, 28, 28, 1))
        self.assertGreaterEqual(processed.min(), 0.0)
        self.assertLessEqual(processed.max(), 1.0)
    
    def test_preprocess_empty_canvas(self):
        """Test preprocessing empty canvas."""
        canvas = self.create_test_canvas(draw_something=False)
        processed = self.preprocess_canvas_image(canvas)
        
        self.assertIsNone(processed)
    
    def test_preprocess_none_input(self):
        """Test preprocessing with None input."""
        processed = self.preprocess_canvas_image(None)
        self.assertIsNone(processed)
    
    def test_preprocess_preserves_drawing(self):
        """Test that preprocessing preserves drawn content."""
        canvas = self.create_test_canvas(draw_something=True)
        processed = self.preprocess_canvas_image(canvas)
        
        # After inversion, drawn areas should have high values
        self.assertGreater(processed.max(), 0.5)
        
        # Should have some variation (not all same value)
        self.assertGreater(processed.std(), 0.01)
    
    def test_label_mapping_exists(self):
        """Test that label mapping file exists."""
        self.assertTrue(os.path.exists(self.label_path))
    
    def test_label_mapping_structure(self):
        """Test label mapping has correct structure."""
        with open(self.label_path, 'r') as f:
            mapping = json.load(f)
        
        # Should have 62 classes
        self.assertEqual(len(mapping), 62)
        
        # Check some expected mappings
        self.assertEqual(mapping['0'], '0')  # Digit 0
        self.assertEqual(mapping['10'], 'A')  # Uppercase A
        self.assertEqual(mapping['36'], 'a')  # Lowercase a
    
    def test_model_file_exists(self):
        """Test that model file exists."""
        self.assertTrue(os.path.exists(self.model_path))
    
    def test_model_file_size(self):
        """Test that model file is reasonable size."""
        if os.path.exists(self.model_path):
            file_size = os.path.getsize(self.model_path)
            # Model should be between 1MB and 100MB
            self.assertGreater(file_size, 1_000_000)
            self.assertLess(file_size, 100_000_000)
            print(f"\nModel size: {file_size / 1_000_000:.2f} MB")
    
    def test_model_loading(self):
        """Test that model can be loaded."""
        if not os.path.exists(self.model_path):
            self.skipTest("Model file not found")
        
        import tensorflow as tf
        try:
            model = tf.keras.models.load_model(self.model_path)
            self.assertIsNotNone(model)
            self.assertEqual(model.input_shape, (None, 28, 28, 1))
            self.assertEqual(model.output_shape, (None, 62))
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    def test_end_to_end_prediction(self):
        """Test complete prediction pipeline."""
        if not os.path.exists(self.model_path):
            self.skipTest("Model file not found")
        
        import tensorflow as tf
        
        # Load model and labels
        model = tf.keras.models.load_model(self.model_path)
        with open(self.label_path, 'r') as f:
            label_mapping = json.load(f)
        
        # Create and preprocess canvas
        canvas = self.create_test_canvas(draw_something=True)
        processed = self.preprocess_canvas_image(canvas)
        
        # Make prediction
        predictions = model.predict(processed, verbose=0)
        
        # Verify prediction shape and values
        self.assertEqual(predictions.shape, (1, 62))
        self.assertAlmostEqual(predictions.sum(), 1.0, places=5)
        
        # Get top prediction
        top_idx = predictions[0].argmax()
        top_char = label_mapping[str(top_idx)]
        top_conf = predictions[0][top_idx]
        
        print(f"\nPrediction: '{top_char}' with {top_conf*100:.1f}% confidence")
        
        self.assertIsInstance(top_char, str)
        self.assertGreater(top_conf, 0.0)
        self.assertLess(top_conf, 1.0)
    
    def test_canvas_size_constant(self):
        """Test that canvas size is as expected."""
        self.assertEqual(self.canvas_size, 280)
    
    def test_preprocessed_image_size(self):
        """Test that preprocessed image is 28x28."""
        canvas = self.create_test_canvas(draw_something=True)
        processed = self.preprocess_canvas_image(canvas)
        self.assertEqual(processed.shape[1:3], (28, 28))


if __name__ == '__main__':
    unittest.main()
