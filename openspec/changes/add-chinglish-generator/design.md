# Design Document: Chinglish Text Generator with Speech

## Context
This project creates an entertaining web application that generates "Chinglish" style text (晶晶體 - the mixed Chinese-English speaking style common among overseas students) and converts it to speech. The application uses pre-downloaded local models deployed on Streamlit to avoid runtime downloads and external API dependencies.

## Goals / Non-Goals

### Goals
- Generate natural-sounding mixed Chinese-English text based on user-provided topics
- Convert generated text to speech with clear pronunciation of both languages
- Provide simple, intuitive Streamlit web interface accessible via browser
- Use pre-downloaded models stored locally to avoid runtime downloads
- Deploy on Streamlit Cloud or local environment
- Support rapid iteration on prompt engineering to improve Chinglish quality

### Non-Goals
- Real-time conversational AI or chatbot functionality
- User authentication or multi-user support
- Text editing or manual correction features
- Mobile native application (web-responsive is sufficient)
- Training custom models (use existing pre-trained models)

## Architecture Overview

```
┌─────────────────────────────────┐
│      Web Browser                │
│   (Streamlit Interface)         │
└────────┬────────────────────────┘
         │ HTTP
         ▼
┌─────────────────────────────────┐
│     Streamlit App               │
│  ┌──────────┐  ┌─────────────┐ │
│  │   HF     │  │     TTS     │ │
│  │Transformers  │  (gTTS)     │ │
│  └─────┬────┘  └──────┬──────┘ │
│        │              │         │
│        ▼              ▼         │
│  ┌──────────────────────────┐  │
│  │   Pre-downloaded Models  │  │
│  │   (models/ directory)    │  │
│  └──────────────────────────┘  │
└─────────────────────────────────┘
```

## Component Decisions

### 1. Language Model Selection

**Decision**: Use Hugging Face transformers with Qwen2-1.5B-Instruct or Qwen2-0.5B-Instruct

**Rationale**:
- Qwen models have excellent Chinese-English bilingual support
- Qwen2-1.5B-Instruct is compact (~3GB) and performs well on instruction following
- Can be pre-downloaded and loaded from local directory
- Works with standard transformers library, no special runtime needed
- Streamlit Cloud compatible with appropriate resource limits
- Can use ONNX runtime for faster inference

**Alternatives Considered**:
- Ollama: Rejected because requires separate service and not Streamlit-compatible
- GPT-4 API: Rejected due to cost and internet dependency
- Larger models (7B+): Rejected due to Streamlit Cloud resource limits
- GPT-2: Rejected due to poor Chinese support

### 2. Text-to-Speech Engine

**Decision**: Use gTTS (Google Text-to-Speech) with caching

**Rationale**:
- Simple Python API that works well with Streamlit
- Good support for both Chinese and English pronunciation
- Can handle mixed-language text by splitting into segments
- Lightweight and easy to integrate
- Generates MP3 files that work with Streamlit audio widget
- Can cache generated audio to avoid repeated API calls

**Alternatives Considered**:
- edge-tts: Requires async, more complex Streamlit integration
- pyttsx3: Lower quality, poor Chinese support, requires system dependencies
- Coqui TTS: Too large for Streamlit deployment
- Azure TTS: Requires API key and costs money

**Fallback**: Provide text-only output if TTS fails.

### 3. Web Framework

**Decision**: Use Streamlit

**Rationale**:
- Built specifically for Python ML/AI applications
- Extremely simple to create interactive web interfaces
- No need for separate frontend/backend code
- Built-in widgets for text input, buttons, audio playback
- Easy deployment to Streamlit Cloud
- Handles state management automatically
- Perfect for demonstration and educational applications

**Alternatives Considered**:
- FastAPI + HTML/JS: More complex, requires separate frontend/backend
- Gradio: Similar to Streamlit but less flexible
- Flask: Requires more boilerplate and frontend work
- Django: Too heavy for this simple use case

### 4. Model Storage and Loading

**Decision**: Pre-download models to `models/` directory

**Rationale**:
- Avoids Hugging Face download delays during app startup or first use
- Works offline once models are downloaded
- More predictable performance on Streamlit Cloud
- Can version control model choice without re-downloading
- Allows use of specific model versions

**Implementation**:
- Use `snapshot_download()` or `from_pretrained(local_files_only=True)`
- Store in `models/qwen2-1.5b-instruct/` directory structure
- Add models to `.gitignore` but document download process
- Provide setup script to download required models

## Key Technical Decisions

### Prompt Engineering Strategy

The system prompt will:
1. Instruct the model to respond as a留学生 (overseas student)
2. Provide examples of natural Chinglish patterns (e.g., "這個project真的很challenging", "我覺得這個approach很make sense")
3. Encourage mixing at phrase level rather than word level
4. Specify common English terms used in Chinese contexts (e.g., deadline, meeting, presentation)

**Example Prompt Template**:
```
你是一個在美國留學的台灣/中國學生，說話時會自然地中英文夾雜（晶晶體風格）。
請根據以下主題生成一段自然的對話或描述，要展現留學生特有的說話方式。

特點：
- 專業術語和概念用英文（如：project, deadline, presentation）
- 日常對話用中文為主
- 自然的code-switching，不要刻意
- 保持句子流暢和自然

主題：{user_topic}

請生成一段100-150字的內容：
```

### Streamlit UI Flow

**User Interface Elements**:
```python
1. Title and description
2. Text input widget for topic
3. Generate button
4. Progress indicator (spinner) during generation
5. Generated text display (st.text_area or st.markdown)
6. Audio player widget (st.audio)
7. Metrics display (generation time, synthesis time)
8. Example topics (st.selectbox or buttons)
```

**Application Flow**:
```
1. User enters topic in text input
2. User clicks "Generate" button
3. App shows spinner with status
4. Load model from models/ directory (cached)
5. Generate Chinglish text with LLM
6. Display generated text
7. Synthesize speech with gTTS
8. Display audio player with generated MP3
9. Show performance metrics
10. Enable new generation
```

### Audio Handling

- Generate audio files and save to `temp_audio/` directory
- Use UUID or timestamp-based filenames to avoid collisions
- Use Streamlit's `st.audio()` widget to display audio player
- Clean up old audio files on app restart or periodically
- Support MP3 format (primary for gTTS compatibility)
- Handle Streamlit Cloud ephemeral storage limitations

### Error Handling

1. **LLM Timeout**: Set 30-second timeout, return friendly error
2. **TTS Failure**: Provide text-only result with error message
3. **Invalid Input**: Validate topic length (max 200 chars), sanitize
4. **Resource Exhaustion**: Queue requests if needed, return 503 if overloaded

## Performance Considerations

### Expected Latency
- Model loading (first time): 10-20 seconds (cached afterwards)
- Text generation: 10-60 seconds (depends on model size and device)
- Speech synthesis: 2-10 seconds (depends on text length and API)
- Total user wait time: 12-70 seconds

### Optimization Strategies
- Use Streamlit's `@st.cache_resource` to cache loaded model
- Use smaller Qwen2-0.5B model if latency is critical
- Consider ONNX runtime for faster inference
- Cache generated audio for repeated topics
- Show progress indicators during long operations
- Pre-download models during deployment setup

### Resource Requirements
- RAM: 4GB minimum for Qwen2-1.5B, 2GB for Qwen2-0.5B
- Storage: 3-4GB (model files + dependencies)
- CPU: Multi-core processor (2+ cores)
- GPU: Optional, but significantly speeds up generation
- Streamlit Cloud: Works with free tier but may be slow

## Security Considerations

1. **Input Sanitization**: Limit topic length, filter malicious content
2. **Rate Limiting**: Prevent abuse with per-IP rate limits
3. **File System**: Sandbox audio file generation to specific directory
4. **CORS**: Configure appropriate origins for production
5. **No Authentication**: Acceptable for local/demo use; add if deploying publicly

## Migration Plan

N/A - This is a new application with no existing users or data.

## Deployment Strategy

### Development
1. Run model download script: `python scripts/download_models.py`
2. Install dependencies: `pip install -r requirements.txt`
3. Start Streamlit: `streamlit run app.py`
4. Access at `http://localhost:8501`

### Production (Streamlit Cloud)
1. Push code to GitHub repository
2. Ensure models/ directory is populated or use download script
3. Configure `packages.txt` for system dependencies
4. Deploy to Streamlit Cloud
5. Models will be included in deployment package

### Alternative (Docker)
1. Build Docker image with pre-downloaded models
2. Use multi-stage build to minimize image size
3. Deploy to any container hosting service

## Open Questions

1. **Model Size**: Use Qwen2-1.5B or Qwen2-0.5B? (Balance quality vs. speed/resources)
2. **TTS Language**: Split Chinese/English text or use single language setting? (Test pronunciation quality)
3. **Prompt Refinement**: What examples produce the most natural Chinglish? (Iterative testing needed)
4. **Caching Strategy**: Cache model in Streamlit session state or use @st.cache_resource? (Performance testing needed)
5. **ONNX Runtime**: Should we use ONNX for faster inference? (Measure speedup vs. complexity)

## Testing Strategy

1. **Unit Tests**: Test prompt generation, text validation, file cleanup
2. **Integration Tests**: Test model loading, text generation, TTS synthesis
3. **Manual QA**: Evaluate Chinglish naturalness and audio quality with native speakers
4. **Performance Tests**: Measure latency with Qwen2-0.5B vs 1.5B models
5. **Error Scenarios**: Test missing models, TTS failures, invalid inputs
6. **Streamlit Testing**: Test state management, caching, UI responsiveness

## Future Enhancements (Out of Scope)

- User accounts and history of generated content
- Adjustable "Chinglish intensity" slider
- Multiple voice options and speed controls
- Export to different audio formats
- Mobile app versions
- Sharing generated audio on social media
- Fine-tuning model specifically for Chinglish style
