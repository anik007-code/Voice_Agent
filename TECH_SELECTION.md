# Technology Selection Justification

## LLM: qwen2:0.5b (Ollama)
- **Reason**: Lightweight, open-source, CPU-friendly model suitable for local development. Provides decent performance for text generation within resource constraints.
- **Trade-offs**: Less powerful than larger models like GPT-3.5, but free and accessible.

## TTS: Coqui TTS (tts_models/en/ljspeech/tacotron2-DDC)
- **Reason**: Open-source, supports CPU execution, and generates clear audio for English. Aligns with the task’s preference for free tools.
- **Trade-offs**: Slower than paid services like ElevenLabs but cost-effective.

## STT: HuggingFace Whisper (openai/whisper-tiny)
- **Reason**: Open-source, efficient for CPU, and widely used for speech recognition. Mock implementation used for simulation, as real telephony isn’t required.
- **Trade-offs**: Tiny model is less accurate but lightweight.

## Framework: LangChain
- **Reason**: Simplifies LLM integration and prompt management. Supports custom LLMs like Ollama.
- **Trade-offs**: Considered LangGraph for state management but chose simpler prompt-based approach for MVP.

## Backend: FastAPI
- **Reason**: Lightweight, supports async operations, and provides automatic OpenAPI documentation.
