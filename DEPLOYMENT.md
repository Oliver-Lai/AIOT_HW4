# Deployment Guide for Streamlit Cloud

This guide explains how to deploy the EMNIST Character Recognition app to Streamlit Cloud.

## Prerequisites

1. **GitHub Account**: Ensure your code is pushed to a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io/)
3. **Model File**: The trained model (`models/emnist_cnn_v1.keras`) must be in the repository

## Pre-Deployment Checklist

- [x] All code committed and pushed to GitHub
- [x] `requirements.txt` contains all dependencies with correct versions
- [x] Model file size < 100MB (current: 20.5 MB) ✅
- [x] `.streamlit/config.toml` configured
- [x] `.gitignore` excludes unnecessary files
- [x] `README.md` is comprehensive
- [x] All tests pass (46/46 tests passing)

## Deployment Steps

### Step 1: Prepare Repository

```bash
# Ensure all changes are committed
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Create Streamlit Cloud Account

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click "Sign up" and authenticate with GitHub
3. Grant Streamlit access to your repositories

### Step 3: Deploy Application

1. Click "New app" button
2. Configure deployment settings:
   - **Repository**: `Oliver-Lai/AIOT_HW4`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **Python version**: `3.12`
3. Click "Deploy!"

### Step 4: Monitor Deployment

1. Watch the build logs for any errors
2. Wait for deployment to complete (typically 2-5 minutes)
3. Once deployed, your app will be accessible at:
   ```
   https://[your-app-name].streamlit.app
   ```

## Expected Build Time

- **First deployment**: 3-5 minutes
- **Subsequent deployments**: 1-2 minutes (cached dependencies)

## Post-Deployment Verification

1. **Test Canvas Drawing**: Draw a character and verify it displays correctly
2. **Test Prediction**: Click "Predict" and verify results appear within 2 seconds
3. **Test Clear Canvas**: Verify the clear button works
4. **Test Multiple Characters**: Try digits, uppercase, and lowercase letters
5. **Check Mobile**: Test on mobile browser (iOS Safari, Chrome Mobile)

## Troubleshooting

### Build Fails

**Issue**: Dependencies fail to install

**Solution**:
- Check `requirements.txt` for version conflicts
- Ensure all packages are available on PyPI
- Try pinning specific versions

### Model Not Found

**Issue**: App shows "Model file not found"

**Solution**:
- Ensure `models/emnist_cnn_v1.keras` is committed to Git
- Check file path is correct in `app.py`
- Verify `.gitignore` doesn't exclude model files

### Out of Memory

**Issue**: App crashes during model loading

**Solution**:
- Verify model size < 100MB (current: 20.5 MB ✅)
- Check TensorFlow version compatibility
- Consider model quantization if needed

### Slow Performance

**Issue**: Predictions take too long

**Solution**:
- Model is cached with `@st.cache_resource`
- First prediction may be slow (model loading)
- Subsequent predictions should be < 100ms

## Configuration Files

### `.streamlit/config.toml`
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
enableCORS = false
port = 8501

[browser]
gatherUsageStats = false
```

### `requirements.txt`
Ensure all dependencies are listed with versions:
```
tensorflow>=2.20.0
streamlit>=1.25.0
streamlit-drawable-canvas>=0.9.0
numpy>=1.23.0
opencv-python-headless>=4.7.0
matplotlib>=3.10.0
scikit-learn>=1.2.0
pandas>=1.5.0
emnist
```

## Performance Targets

- ✅ **Model Size**: 20.5 MB (< 100 MB limit)
- ✅ **Cold Start**: < 7 seconds
- ✅ **Warm Prediction**: < 100 ms (measured: 49.7 ms)
- ✅ **Model Accuracy**: ≥ 85% on test set

## Support

If you encounter issues:
1. Check [Streamlit Community Forum](https://discuss.streamlit.io/)
2. Review [Streamlit Documentation](https://docs.streamlit.io/)
3. Open an issue on the GitHub repository

## Updating the Deployment

After making changes:
```bash
git add .
git commit -m "Update: [description]"
git push origin main
```

Streamlit Cloud will automatically detect changes and redeploy (auto-deploy enabled by default).

## Custom Domain (Optional)

To use a custom domain:
1. Go to app settings in Streamlit Cloud
2. Navigate to "General" tab
3. Add custom domain
4. Configure DNS records as instructed

---

Last Updated: December 10, 2025
