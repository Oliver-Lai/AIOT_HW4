"""
EMNIST Handwritten Character Recognition - Streamlit Application

This application allows users to draw characters and get real-time predictions
using a trained CNN model on the EMNIST dataset.
"""

import streamlit as st
import numpy as np
import json
import os
import cv2
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from PIL import Image

# Constants
MODEL_PATH = "models/emnist_cnn_v1.keras"
LABEL_MAPPING_PATH = "models/label_mapping.json"
CANVAS_SIZE = 280
IMAGE_SIZE = 28

# Page configuration
st.set_page_config(
    page_title="EMNIST Character Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Helper Functions ====================

@st.cache_resource
def load_model():
    """Load the trained CNN model with caching."""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            st.info("Please train a model first using the training scripts.")
            return None
        
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

@st.cache_data
def load_label_mapping():
    """Load the label mapping from JSON file."""
    try:
        if not os.path.exists(LABEL_MAPPING_PATH):
            st.error(f"❌ Label mapping file not found at: {LABEL_MAPPING_PATH}")
            return None
        
        with open(LABEL_MAPPING_PATH, 'r') as f:
            mapping = json.load(f)
        return mapping
    except Exception as e:
        st.error(f"❌ Error loading label mapping: {str(e)}")
        return None

def preprocess_canvas_image(canvas_data):
    """
    Preprocess canvas image for model prediction.
    
    Args:
        canvas_data: RGBA image data from streamlit-drawable-canvas
    
    Returns:
        Preprocessed image array ready for prediction, or None if canvas is empty
    """
    if canvas_data is None:
        return None
    
    # Convert RGBA to grayscale
    # The canvas has black strokes on white background
    # Use RGB channels (ignore alpha) and convert to grayscale
    img_rgb = canvas_data[:, :, :3]  # Get RGB channels
    img_gray = cv2.cvtColor(img_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Check if canvas is empty (all white means no drawing)
    if img_gray.min() > 250:  # If minimum value is very high, canvas is empty
        return None
    
    # Resize to 28x28 using area interpolation
    img_resized = cv2.resize(img_gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    # Invert colors: canvas is black-on-white, but EMNIST is white-on-black
    img_inverted = 255 - img_resized
    
    # Normalize to [0, 1]
    img_normalized = img_inverted.astype(np.float32) / 255.0
    
    # Reshape to (1, 28, 28, 1) for model input
    img_final = img_normalized.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)
    
    return img_final

def predict_character(model, image, label_mapping, top_k=5):
    """
    Predict character from preprocessed image.
    
    Args:
        model: Trained Keras model
        image: Preprocessed image array (1, 28, 28, 1)
        label_mapping: Dictionary mapping indices to characters
        top_k: Number of top predictions to return
    
    Returns:
        List of (character, confidence) tuples
    """
    try:
        # Get predictions
        predictions = model.predict(image, verbose=0)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            char = label_mapping[str(idx)]
            confidence = float(predictions[idx]) * 100
            results.append((char, confidence))
        
        return results
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# Application header
st.title("✍️ EMNIST Handwritten Character Recognition")
st.markdown("""
This application uses a Convolutional Neural Network (CNN) trained on the EMNIST dataset 
to recognize handwritten English letters (uppercase and lowercase) and digits (0-9).
""")

# Usage instructions
with st.expander("📖 How to Use", expanded=True):
    st.markdown("""
    ### Instructions:
    1. **Draw** a character on the canvas using your mouse or touchscreen
    2. **Adjust** the stroke width if needed using the slider
    3. **Click** the "Predict" button to see the model's prediction
    4. **Clear** the canvas to try another character
    
    ### Tips:
    - Draw clearly and use most of the canvas space
    - Single characters work best (avoid drawing multiple characters)
    - The model recognizes: **Digits (0-9)**, **Uppercase (A-Z)**, and **Lowercase (a-z)**
    - Try adjusting stroke width for better results
    """)

# ==================== Load Model and Mapping ====================

with st.spinner("🔄 Loading model..."):
    model = load_model()
    label_mapping = load_label_mapping()

if model is None or label_mapping is None:
    st.stop()

# ==================== Main Application ====================

# Initialize session state for canvas reset
if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0

# Create layout: canvas on left, results on right
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✏️ Draw Here")
    
    # Stroke width slider
    stroke_width = st.slider("Stroke Width", min_value=5, max_value=30, value=15, step=1)
    
    # Drawing canvas (with dynamic key for reset functionality)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
        stroke_width=stroke_width,
        stroke_color="#000000",  # Black stroke
        background_color="#FFFFFF",  # White background
        height=CANVAS_SIZE,
        width=CANVAS_SIZE,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}",
    )
    
    # Action buttons
    col_predict, col_clear = st.columns(2)
    with col_predict:
        predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.session_state.canvas_key += 1
            st.rerun()

with col2:
    st.subheader("📊 Predictions")
    
    # Handle prediction
    if predict_button:
        if canvas_result.image_data is not None:
            # Preprocess image
            processed_image = preprocess_canvas_image(canvas_result.image_data)
            
            if processed_image is None:
                st.warning("⚠️ Canvas is empty! Please draw a character first.")
            else:
                # Show preprocessing preview
                with st.spinner("🔄 Analyzing..."):
                    # Make prediction
                    results = predict_character(model, processed_image, label_mapping, top_k=5)
                    
                    if results:
                        # Display results
                        st.success("✅ Prediction Complete!")
                        
                        # Top prediction (highlighted)
                        top_char, top_conf = results[0]
                        
                        # Show warning if confidence is low
                        if top_conf < 50:
                            st.warning("⚠️ Low confidence prediction. Try drawing more clearly.")
                        
                        # Display top prediction prominently
                        st.markdown(f"""
                        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
                            <h1 style='margin: 0; font-size: 72px; color: #1f77b4;'>{top_char}</h1>
                            <p style='margin: 5px 0 0 0; font-size: 24px; color: #666;'>{top_conf:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display all top-5 predictions
                        st.markdown("**Top 5 Predictions:**")
                        for i, (char, conf) in enumerate(results, 1):
                            if i == 1:
                                st.markdown(f"**{i}. `{char}` - {conf:.1f}%** 🏆")
                            else:
                                st.markdown(f"{i}. `{char}` - {conf:.1f}%")
                        
                        # Show preprocessed image
                        with st.expander("🔍 View Preprocessed Image"):
                            resized_for_display = cv2.resize(
                                processed_image[0, :, :, 0], 
                                (140, 140), 
                                interpolation=cv2.INTER_NEAREST
                            )
                            st.image(resized_for_display, caption="28x28 Model Input", clamp=True)
        else:
            st.info("👆 Draw a character and click Predict!")
    else:
        st.info("👆 Draw a character on the canvas and click the **Predict** button to see the results!")

# ==================== Sidebar: Model Information ====================

with st.sidebar:
    st.header("ℹ️ About")
    
    with st.expander("🤖 Model Information", expanded=False):
        st.markdown("""
        **Architecture:** Convolutional Neural Network (CNN)
        
        **Dataset:** EMNIST ByClass
        - Training samples: ~697,932
        - Test samples: ~116,323
        - Classes: 62 (digits + uppercase + lowercase)
        
        **Model Details:**
        - 3 Convolutional blocks (64, 128, 256 filters)
        - Batch normalization
        - MaxPooling layers
        - 2 Dense layers (512, 256 units)
        - Dropout regularization
        - Total parameters: ~1.7M
        
        **Expected Accuracy:** ≥85% on test set
        """)
    
    with st.expander("📝 Supported Characters", expanded=False):
        st.markdown("""
        **Digits (10):**
        ```
        0 1 2 3 4 5 6 7 8 9
        ```
        
        **Uppercase Letters (26):**
        ```
        A B C D E F G H I J K L M
        N O P Q R S T U V W X Y Z
        ```
        
        **Lowercase Letters (26):**
        ```
        a b c d e f g h i j k l m
        n o p q r s t u v w x y z
        ```
        
        **Total: 62 classes**
        """)
    
    st.divider()
    
    # Tips
    st.markdown("### 💡 Tips for Best Results")
    st.markdown("""
    - Draw characters **large and clear**
    - Use the **center of the canvas**
    - Keep strokes **connected**
    - Try different **stroke widths**
    - One character at a time
    """)

# Footer
st.divider()
st.caption("Built with Streamlit and TensorFlow • EMNIST Dataset • CNN Model")
