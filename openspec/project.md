# Project Context

## Purpose
Build an EMNIST handwritten character recognition system that trains a deep learning model and deploys it as an interactive Streamlit web application for real-time character prediction.

## Tech Stack
- **Deep Learning**: TensorFlow/Keras (primary) or PyTorch
- **Web Framework**: Streamlit
- **UI Components**: streamlit-drawable-canvas
- **Image Processing**: OpenCV or Pillow
- **Data Science**: NumPy, Pandas, Matplotlib, scikit-learn
- **Deployment**: Streamlit Cloud
- **Version Control**: Git/GitHub

## Project Conventions

### Code Style
- Python 3.8+ with PEP 8 style guidelines
- Type hints for function signatures where appropriate
- Docstrings for all modules, classes, and functions
- Descriptive variable names (e.g., `preprocessed_image` not `img2`)
- Maximum line length: 100 characters

### Architecture Patterns
- Modular design with clear separation: data, models, training, utilities, web app
- Configuration over hardcoding (use constants at module level)
- Lazy loading for heavy resources (models) in web application
- Stateless preprocessing functions for testability
- Single responsibility principle for functions and classes

### Testing Strategy
- Unit tests for data preprocessing and utility functions
- Integration tests for end-to-end prediction pipeline
- Manual testing for web interface interactions
- Target: >80% code coverage for core modules
- Use pytest as testing framework

### Git Workflow
- Main branch for stable, deployable code
- Feature branches for development (feature/model-training, feature/web-ui)
- Descriptive commit messages with imperative mood
- Commit trained models with Git LFS if >50MB
- .gitignore excludes: data/, __pycache__/, *.pyc, venv/, .DS_Store

## Domain Context
- **EMNIST Dataset**: Extended MNIST with 62 classes (digits, uppercase, lowercase letters)
- **Character Recognition**: Computer vision task requiring image preprocessing and CNN classification
- **Streamlit**: Python framework for rapid web app development, optimized for ML/data science
- **Model Deployment**: Focus on small, efficient models (<100MB) for cloud deployment
- **Interactive ML**: Real-time inference with <2 second latency requirement

## Important Constraints
- Model size must be <100MB for Streamlit Cloud deployment
- Streamlit Cloud free tier: 1GB RAM, limited CPU
- Target accuracy: ≥85% on EMNIST test set
- Inference latency: <2 seconds total (preprocessing + prediction + display)
- Browser compatibility: Modern browsers (Chrome, Firefox, Safari)
- No user authentication or data persistence required

## External Dependencies
- **EMNIST Dataset**: Available via torchvision.datasets or tensorflow_datasets
- **Streamlit Cloud**: Hosting platform for web application
- **GitHub**: Repository hosting and version control
- **Python Package Index (PyPI)**: All dependencies installable via pip
