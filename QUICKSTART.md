# Quick Start Guide - EMNIST Character Recognition App

This guide will help you get the Streamlit application up and running quickly.

## Prerequisites

✅ Python 3.8 or higher
✅ All dependencies installed (`pip install -r requirements.txt`)
✅ Model trained and saved (or use the provided model)

## Starting the Application

### Method 1: Using Streamlit Command (Recommended)

```bash
streamlit run app.py
```

The app will automatically open in your default browser at `http://localhost:8501`.

### Method 2: Specify Port

```bash
streamlit run app.py --server.port 8502
```

### Method 3: Headless Mode (for servers)

```bash
streamlit run app.py --server.headless true
```

## Using the Application

### Step 1: Draw a Character
- Use your mouse or touchscreen to draw on the white canvas
- Draw clearly and use most of the canvas space
- Single characters work best

### Step 2: Adjust Stroke Width (Optional)
- Use the slider to adjust pen thickness (5-30 pixels)
- Larger strokes may work better for touchscreens
- Smaller strokes give more precise control

### Step 3: Predict
- Click the "🔮 Predict" button
- Wait for the model to analyze your drawing (usually <1 second)
- View the top-5 predictions with confidence scores

### Step 4: Try Another Character
- Click "🗑️ Clear Canvas" to start fresh
- Draw a new character and repeat

## Features

### Main Canvas Area
- **280×280 pixel canvas** - Large enough for detailed drawing
- **Stroke width slider** - Adjust pen size for your preference
- **Real-time drawing** - Smooth drawing experience
- **Clear button** - Quickly reset the canvas

### Prediction Results
- **Top prediction highlighted** - Large display of the most likely character
- **Confidence scores** - See how certain the model is
- **Top-5 alternatives** - View other possible matches
- **Low confidence warning** - Alert when prediction may be unreliable
- **Preprocessed image preview** - See what the model actually sees

### Sidebar Information
- **Model Information** - Architecture and dataset details
- **Supported Characters** - Complete list of 62 recognizable characters
- **Tips for Best Results** - Helpful suggestions

## Supported Characters

The model can recognize **62 different characters**:

- **Digits (10):** 0-9
- **Uppercase Letters (26):** A-Z
- **Lowercase Letters (26):** a-z

## Tips for Best Results

✨ **Draw clearly** - Make characters neat and easy to read
✨ **Use the full canvas** - Don't make characters too small
✨ **Keep strokes connected** - Avoid lifting your pen/finger too much
✨ **Center your drawing** - Keep the character in the middle of the canvas
✨ **Try different stroke widths** - Find what works best for you
✨ **One character at a time** - The model is trained on single characters

## Troubleshooting

### App Won't Start

**Problem:** `ModuleNotFoundError`
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

**Problem:** `Model file not found`
**Solution:** Train a model first
```bash
python train_quick_model.py
```

### Poor Predictions

**Problem:** Model always predicts incorrectly
**Solution:**
- Draw more clearly and larger
- Try adjusting stroke width
- Make sure you're drawing only one character
- Check that the character is centered

**Problem:** Low confidence warnings
**Solution:**
- The character may be ambiguous (like 'O' vs '0')
- Try drawing more clearly
- Ensure the entire character is visible

### Canvas Issues

**Problem:** Can't draw on canvas
**Solution:**
- Ensure JavaScript is enabled in your browser
- Try refreshing the page (Ctrl/Cmd + R)
- Try a different browser (Chrome, Firefox, Safari recommended)

**Problem:** Clear button doesn't work
**Solution:**
- The page will automatically refresh when you click Clear
- If it doesn't, manually refresh the page

## Performance Notes

- **First prediction is slower** - Model loads on first use (~2-5 seconds)
- **Subsequent predictions are fast** - Usually <1 second
- **Model is cached** - No need to reload between predictions
- **CPU mode** - App works without GPU (may be slightly slower)

## Keyboard Shortcuts

While the app is running, you can use Streamlit's built-in shortcuts:

- `Ctrl/Cmd + R` - Refresh the app
- `Ctrl/Cmd + K` - Open command palette
- `Ctrl/Cmd + /` - Show keyboard shortcuts

## Advanced Options

### Running with Custom Settings

```bash
# Change port
streamlit run app.py --server.port 8080

# Disable CORS (for remote access)
streamlit run app.py --server.enableCORS false

# Increase upload size limit
streamlit run app.py --server.maxUploadSize 200

# Run with specific config file
streamlit run app.py --config .streamlit/config.toml
```

### Accessing from Other Devices

1. Find your local IP address:
   ```bash
   # Linux/Mac
   ifconfig | grep "inet "
   
   # Windows
   ipconfig
   ```

2. Open the app at `http://YOUR_IP:8501` on other devices on the same network

## Next Steps

- ✅ Draw different characters and test accuracy
- ✅ Compare uppercase vs lowercase recognition
- ✅ Test commonly confused pairs (like 'O' vs '0', 'I' vs 'l')
- ✅ Try challenging characters (like 'Q', 'q', 'g', etc.)
- 📊 Review evaluation metrics in `notebooks/04_model_evaluation.ipynb`
- 🚀 Deploy to Streamlit Cloud (see deployment guide)

## Getting Help

If you encounter issues:

1. **Run tests:** `python test_app.py` to verify everything works
2. **Check logs:** Streamlit shows errors in the terminal
3. **Review documentation:** See `README.md` for more details
4. **Model evaluation:** Run `python src/training/evaluate.py` to check model performance

## Stopping the Application

Press `Ctrl + C` in the terminal where Streamlit is running to stop the app.

---

**Enjoy using the EMNIST Character Recognition App!** 🎉
