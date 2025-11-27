# Change: Add Chinglish Style Text Generator with Speech Output

## Why
Students who study abroad often develop a communication style mixing Chinese and English (commonly called 晶晶體 or "Chinglish"). This application provides an entertaining and educational tool to generate text in this distinctive style using a local lightweight language model, then convert it to speech for listening playback.

## What Changes
- Add text generation capability using pre-downloaded local lightweight LLM (Hugging Face transformers with Qwen or similar bilingual models)
- Add "Chinglish" style prompt engineering to generate mixed Chinese-English conversational text
- Add text-to-speech synthesis capability supporting both Chinese and English characters
- Add Streamlit web interface allowing users to input topics and receive generated audio
- Pre-download and cache model files locally to avoid runtime downloads
- Implement audio playback directly in Streamlit interface

## Impact
- New capabilities: `text-generation`, `speech-synthesis`, `web-interface`
- New dependencies: Streamlit, Hugging Face transformers, torch/onnxruntime, gTTS or pyttsx3
- Deployment: Streamlit Cloud compatible, models stored in local directory
- Performance considerations: Text generation may take 10-60 seconds depending on model size and hardware
- Storage: Requires 2-4GB for pre-downloaded model files

## Technical Approach
- **LLM Backend**: Use Hugging Face transformers with pre-downloaded Qwen2-1.5B-Instruct or similar lightweight bilingual models
- **Model Storage**: Download models to `models/` directory during setup, load from disk at runtime
- **Prompt Engineering**: Design system prompts that encourage mixing Chinese and English naturally
- **TTS Integration**: Use gTTS (Google TTS) or pyttsx3 for speech synthesis
- **Web Framework**: Streamlit for simple, interactive web interface
- **Audio Format**: MP3 or WAV output with Streamlit audio player widget
