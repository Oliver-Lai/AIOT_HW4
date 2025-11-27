# Project Context

## Purpose
AIOT HW4: A web application that generates "Chinglish" (晶晶體) style text using local language models and converts it to speech. This project demonstrates integration of local LLM inference with text-to-speech synthesis for entertaining and educational purposes.

## Tech Stack
- **Web Framework**: Streamlit (Python-based web application framework)
- **LLM**: Hugging Face transformers with Qwen2-1.5B-Instruct (pre-downloaded local model)
- **TTS**: gTTS (Google Text-to-Speech)
- **Model Storage**: Pre-downloaded models in `models/` directory
- **Audio**: MP3 format with Streamlit audio player widget
- **Development Environment**: Ubuntu 24.04 in dev container

## Project Conventions

### Code Style
- Python: PEP 8 style guide, use type hints
- JavaScript: Modern ES6+ syntax, async/await for API calls
- File naming: snake_case for Python, kebab-case for web assets
- Configuration: Use environment variables for settings
- Comments: Docstrings for functions, inline for complex logic

### Architecture Patterns
- Streamlit application with single-file implementation (app.py)
- Separation of concerns: Model loading, text generation, TTS synthesis as separate functions
- Model caching using @st.cache_resource decorator
- Error handling with try-except blocks and user-friendly error display
- Progress feedback using st.spinner for long operations
- Modular functions that can be tested independently

### Testing Strategy
- Manual testing for LLM output quality (Chinglish naturalness)
- Integration tests for API endpoints
- Error scenario testing (timeouts, service failures)
- Cross-browser audio playback validation
- Performance testing for concurrent requests

### Git Workflow
- Main branch for stable code
- Feature branches for new capabilities
- Descriptive commit messages
- OpenSpec-driven development workflow

## Domain Context

### Chinglish (晶晶體) Style
- Communication style common among overseas students
- Natural code-switching between Chinese and English
- Technical terms typically in English (project, deadline, meeting)
- Conversational phrases in Chinese
- Phrase-level mixing rather than word-level
- Examples: "這個project真的很challenging", "我覺得這個approach很make sense"

### Target Users
- Mandarin Chinese speakers familiar with overseas student culture
- Entertainment and educational use cases
- Local deployment (no cloud dependencies)

## Important Constraints
- **Pre-downloaded models**: Models must be downloaded before deployment, no runtime downloads
- **Streamlit Cloud compatible**: Must work within Streamlit Cloud resource limits
- **Resource limits**: Must run with 4GB RAM (Qwen2-1.5B) or 2GB (Qwen2-0.5B)
- **Latency**: Text generation 10-60s, TTS 2-10s is acceptable
- **Language support**: Must handle both simplified and traditional Chinese
- **Audio quality**: Clear pronunciation of both languages required

## External Dependencies
- Streamlit: Web application framework (pip installable)
- Hugging Face transformers: For loading pre-trained models
- PyTorch or ONNX Runtime: Model inference engine
- gTTS: Python package for TTS (requires internet for synthesis)
- Modern web browser: For Streamlit interface

## System Requirements
- OS: Linux (Ubuntu recommended), macOS, or Windows
- RAM: 4GB minimum for Qwen2-1.5B, 2GB for Qwen2-0.5B
- Storage: 3-4GB (model files + dependencies)
- CPU: Multi-core processor (2+ cores)
- GPU: Optional, significantly speeds up generation
- Internet: Required for gTTS synthesis (not for model inference)
