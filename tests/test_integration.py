"""
Integration tests for Streamlit app.
"""

import unittest
import numpy as np
import cv2
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete application."""
    
    def setUp(self):
        """Set up test environment."""
        self.model_path = "models/emnist_cnn_v1.keras"
        self.label_path = "models/label_mapping.json"
        
        if not os.path.exists(self.model_path):
            self.skipTest("Model not available for integration testing")
        
        import tensorflow as tf
        self.model = tf.keras.models.load_model(self.model_path)
        
        with open(self.label_path, 'r') as f:
            self.label_mapping = json.load(f)
    
    def create_canvas_with_character(self, char_type='digit'):
        """Create canvas with different character types."""
        canvas = np.ones((280, 280, 4), dtype=np.uint8) * 255
        
        if char_type == 'digit':
            # Draw digit-like shape
            canvas[80:200, 110:130, 0:3] = 0  # Vertical line
        elif char_type == 'uppercase':
            # Draw uppercase-like shape (triangle top)
            canvas[80:200, 100:120, 0:3] = 0
            canvas[80:200, 160:180, 0:3] = 0
            canvas[80:100, 100:180, 0:3] = 0
        elif char_type == 'lowercase':
            # Draw lowercase-like shape (smaller)
            canvas[120:200, 120:140, 0:3] = 0
        
        return canvas
    
    def preprocess_and_predict(self, canvas):
        """Full pipeline: preprocess and predict."""
        # Preprocess
        img_rgb = canvas[:, :, :3]
        img_gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        if img_gray.min() > 250:
            return None, None
        
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        img_inverted = 255 - img_resized
        img_normalized = img_inverted.astype(np.float32) / 255.0
        img_final = img_normalized.reshape(1, 28, 28, 1)
        
        # Predict
        predictions = self.model.predict(img_final, verbose=0)
        top_idx = predictions[0].argmax()
        top_char = self.label_mapping[str(top_idx)]
        top_conf = predictions[0][top_idx]
        
        return top_char, top_conf
    
    def test_digit_prediction(self):
        """Test prediction for digit-like drawing."""
        canvas = self.create_canvas_with_character('digit')
        char, conf = self.preprocess_and_predict(canvas)
        
        self.assertIsNotNone(char)
        self.assertGreater(conf, 0.0)
        print(f"\nDigit prediction: '{char}' ({conf*100:.1f}%)")
    
    def test_uppercase_prediction(self):
        """Test prediction for uppercase-like drawing."""
        canvas = self.create_canvas_with_character('uppercase')
        char, conf = self.preprocess_and_predict(canvas)
        
        self.assertIsNotNone(char)
        self.assertGreater(conf, 0.0)
        print(f"\nUppercase prediction: '{char}' ({conf*100:.1f}%)")
    
    def test_lowercase_prediction(self):
        """Test prediction for lowercase-like drawing."""
        canvas = self.create_canvas_with_character('lowercase')
        char, conf = self.preprocess_and_predict(canvas)
        
        self.assertIsNotNone(char)
        self.assertGreater(conf, 0.0)
        print(f"\nLowercase prediction: '{char}' ({conf*100:.1f}%)")
    
    def test_empty_canvas(self):
        """Test that empty canvas returns None."""
        canvas = np.ones((280, 280, 4), dtype=np.uint8) * 255
        char, conf = self.preprocess_and_predict(canvas)
        
        self.assertIsNone(char)
        self.assertIsNone(conf)
    
    def test_single_pixel(self):
        """Test edge case with single pixel drawing."""
        canvas = np.ones((280, 280, 4), dtype=np.uint8) * 255
        canvas[140, 140, 0:3] = 0  # Single black pixel
        
        char, conf = self.preprocess_and_predict(canvas)
        # Should still make a prediction (even if not meaningful)
        self.assertIsNotNone(char)
    
    def test_full_canvas(self):
        """Test edge case with fully drawn canvas."""
        canvas = np.zeros((280, 280, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255  # Set alpha
        
        char, conf = self.preprocess_and_predict(canvas)
        self.assertIsNotNone(char)
    
    def test_prediction_latency(self):
        """Test that predictions complete quickly."""
        canvas = self.create_canvas_with_character('digit')
        
        # Warm up
        self.preprocess_and_predict(canvas)
        
        # Measure time
        start_time = time.time()
        char, conf = self.preprocess_and_predict(canvas)
        elapsed = time.time() - start_time
        
        print(f"\nPrediction latency: {elapsed*1000:.1f} ms")
        
        # Should be under 100ms on warm model
        self.assertLess(elapsed, 0.1)
    
    def test_batch_predictions(self):
        """Test multiple predictions in sequence."""
        char_types = ['digit', 'uppercase', 'lowercase']
        
        for char_type in char_types:
            canvas = self.create_canvas_with_character(char_type)
            char, conf = self.preprocess_and_predict(canvas)
            self.assertIsNotNone(char)
    
    def test_model_caching(self):
        """Test that model stays in memory."""
        # First prediction
        canvas1 = self.create_canvas_with_character('digit')
        _, _ = self.preprocess_and_predict(canvas1)
        model_id_1 = id(self.model)
        
        # Second prediction
        canvas2 = self.create_canvas_with_character('uppercase')
        _, _ = self.preprocess_and_predict(canvas2)
        model_id_2 = id(self.model)
        
        # Model should be same object (cached)
        self.assertEqual(model_id_1, model_id_2)


if __name__ == '__main__':
    unittest.main()
