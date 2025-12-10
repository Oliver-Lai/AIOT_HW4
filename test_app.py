"""
Test script to verify Streamlit app components work correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import streamlit as st
        import numpy as np
        import json
        import cv2
        from streamlit_drawable_canvas import st_canvas
        import tensorflow as tf
        from PIL import Image
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_loading():
    """Test that model and label mapping can be loaded."""
    print("\nTesting model loading...")
    try:
        import tensorflow as tf
        import json
        
        model_path = "models/emnist_cnn_v1.keras"
        label_path = "models/label_mapping.json"
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found at {model_path}")
            print("   Run training first: python train_quick_model.py")
            return False
        
        if not os.path.exists(label_path):
            print(f"❌ Label mapping not found at {label_path}")
            return False
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        
        # Load label mapping
        with open(label_path, 'r') as f:
            label_mapping = json.load(f)
        print(f"✅ Label mapping loaded: {len(label_mapping)} classes")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_preprocessing():
    """Test preprocessing function."""
    print("\nTesting preprocessing function...")
    try:
        import numpy as np
        import cv2
        
        # Create dummy canvas data (RGBA image) - white background
        canvas_size = 280
        dummy_canvas = np.ones((canvas_size, canvas_size, 4), dtype=np.uint8) * 255
        
        # Draw a black square (simulating user drawing)
        dummy_canvas[100:180, 100:180, 0:3] = 0  # Black in RGB channels
        
        # Preprocess (same as app.py)
        img_rgb = dummy_canvas[:, :, :3]
        img_gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        img_inverted = 255 - img_resized
        img_normalized = img_inverted.astype(np.float32) / 255.0
        img_final = img_normalized.reshape(1, 28, 28, 1)
        
        print(f"✅ Preprocessing successful")
        print(f"   Input shape: {dummy_canvas.shape}")
        print(f"   Output shape: {img_final.shape}")
        print(f"   Value range: [{img_final.min():.3f}, {img_final.max():.3f}]")
        print(f"   Non-zero pixels: {(img_final > 0.1).sum()}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_inference():
    """Test model inference."""
    print("\nTesting model inference...")
    try:
        import tensorflow as tf
        import numpy as np
        import json
        
        model_path = "models/emnist_cnn_v1.keras"
        label_path = "models/label_mapping.json"
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model not found, skipping inference test")
            return False
        
        # Load model and labels
        model = tf.keras.models.load_model(model_path)
        with open(label_path, 'r') as f:
            label_mapping = json.load(f)
        
        # Create dummy input
        dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)
        
        # Make prediction
        predictions = model.predict(dummy_input, verbose=0)
        
        # Get top-5
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        
        print(f"✅ Inference successful")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Top 5 predictions:")
        for idx in top_indices:
            char = label_mapping[str(idx)]
            conf = predictions[0][idx] * 100
            print(f"      {char}: {conf:.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("STREAMLIT APP VERIFICATION TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_loading,
        test_preprocessing,
        test_inference,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! App is ready to use.")
        print("\nRun the app with: streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
