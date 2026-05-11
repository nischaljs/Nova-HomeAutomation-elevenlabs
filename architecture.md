# Nova Home Automation — Architecture

## Two Hardware Components

| Component | Role |
|-----------|------|
| **ESP32** (w/ INMP441 mic + I2S speaker) | Trigger-word detection (neural net), audio streaming to Pi, audio playback from Pi |
| **Raspberry Pi** (this codebase) | Server: VAD, STT, face recognition, LLM, TTS, orchestration |

---

## Data Flow (Human Speaking → Audio Response)

```
Human speaks
  │
  ▼
┌─────────────────────────────────────────────────┐
│  ESP32                                           │
│                                                  │
│  1. KWS always running:                          │
│     capture_samples() → audio_inference_callback()│
│     → runSlice() → run_classifier_continuous()   │
│     (Edge Impulse neural net)                    │
│                                                  │
│  2. Trigger "hey nova" > 0.9 CONFIDENCE_THRESHOLD│
│     → startStream()                              │
│     → stops KWS, streams raw 16-bit PCM          │
│       via WebSocket binary frames (320 samples)  │
│                                                  │
│  3. ESP32 has a 10-second streaming timeout:     │
│     if (millis() - streamStart > 10000)           │
│       stopStream()  ← fallback, not VAD          │
│                                                  │
│  WebSocket ──── binary audio ──────────────────► │
└──────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│  RASPBERRY PI SERVER                             │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  app/communication/websocket.py           │   │
│  │  handle_client() receives binary frames   │   │
│  └──────────────┬───────────────────────────┘   │
│                 │                                │
│                 ▼                                │
│  ┌──────────────────────────────────────────┐   │
│  │  app/vad/vad_detection.py                │   │
│  │  detect_speech(audio_data)               │   │
│  │  - WebRTC VAD (aggressiveness=3)         │   │
│  │  - Volume threshold > 600                │   │
│  │  - 5 frames to confirm speech start      │   │
│  │  - 4 seconds silence → speech ends       │   │
│  │    SILENCE_LIMIT = 4000ms / 20ms = 200   │   │
│  │  - 6 seconds no-speech → kill stream     │   │
│  │    NO_SPEECH_LIMIT = 6000ms / 20ms = 300 │   │
│  │  - Saves buffered audio to speech.wav    │   │
│  │    via save_audio() in app/audio/utils.py │   │
│  │  - Sends "stop_stream" to ESP            │   │
│  └──────────────┬───────────────────────────┘   │
│                 │                                │
│  NOTE: VAD is on the Pi ❗                      │
│  The ESP32 does NOT do silence detection.        │
│  It blindly streams until told "stop_stream"     │
│  (or hits its 10-second fallback timeout).       │
│                 │                                │
│                 ▼                                │
│  ┌──────────────────────────────────────────┐   │
│  │  app/pipeline/process_audio.py           │   │
│  │  process_audio()                         │   │
│  │                                           │   │
│  │  Runs two tasks IN PARALLEL:             │   │
│  │  ┌─────────────────┐ ┌────────────────┐  │   │
│  │  │ STT via Groq     │ │ Face Rec       │  │   │
│  │  │ transcribe_audio │ │ _recognize_    │  │   │
│  │  │ (whisper-large-  │ │ snapshot()     │  │   │
│  │  │  v3)              │ │ → get_bridge() │  │   │
│  │  │ speech.wav → text │ │   .recognize() │  │   │
│  │  └────────┬─────────┘ └───────┬────────┘  │   │
│  │           │                    │            │   │
│  │           └───────┬────────────┘            │   │
│  │                   ▼                         │   │
│  │  update_face_context(face_info)             │   │
│  │  → build_face_context() or                  │   │
│  │    build_unknown_context()                  │   │
│  │  → PersonMemory (history, notes, visits)    │   │
│  │                                            │   │
│  │                   ▼                         │   │
│  │  ┌────────────────────────────────────┐    │   │
│  │  │  app/llm/groq.py                  │    │   │
│  │  │  groq_llm_json(user_text)         │    │   │
│  │  │  - Builds system prompt with      │    │   │
│  │  │    face context + intent rules    │    │   │
│  │  │  - Calls Groq API                │    │   │
│  │  │    (openai/gpt-oss-120b)         │    │   │
│  │  │  - Returns JSON with type:       │    │   │
│  │  │    "command" | "query" |         │    │   │
│  │  │    "register"                    │    │   │
│  │  │  - Saves exchange to memory via  │    │   │
│  │  │    PersonMemory.add_history()    │    │   │
│  │  └──────────────┬───────────────────┘    │   │
│  │                 │                          │   │
│  │          ┌──────┼──────────┐               │   │
│  │          ▼      ▼          ▼               │   │
│  │   ┌────────┐┌───────┐┌──────────┐        │   │
│  │   │ query  ││command││ register │        │   │
│  │   │ TTS    ││music/ ││ register │        │   │
│  │   │ reply  ││device ││_face()   │        │   │
│  │   │        ││action ││(save     │        │   │
│  │   │        ││       ││ face DB) │        │   │
│  │   └───┬────┘└───┬───┘└──────────┘        │   │
│  │       │         │                          │   │
│  │       ▼         ▼                          │   │
│  │  ┌────────┐ ┌──────────┐                  │   │
│  │  │ TTS    │ │ Broadcast │                  │   │
│  │  │ Kokoro │ │ to ESP    │                  │   │
│  │  │ (English│ │ (light    │                  │   │
│  │  │ or Edge│ │  on/off)  │                  │   │
│  │  │ -TTS   │ │           │                  │   │
│  │  │ for    │ │           │                  │   │
│  │  │ Nepali)│ │           │                  │   │
│  │  │ + play │ │           │                  │   │
│  │  └───┬────┘ └──────────┘                  │   │
│  │       │                                     │   │
│  │       ▼                                     │   │
│  │  send_websocket_message("start_stream")      │   │
│  │  → ESP32 stopStream() → startKWS()           │   │
│  └─────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│  ESP32 (response path)                          │
│                                                  │
│  Receives "start_stream" → startStream()        │
│    → starts streaming mic audio to Pi again     │
│                                                  │
│  OR receives device command:                     │
│    {"type":"command","payload":{"light":"on"}}   │
│    → handleCommandMessage() → digitalWrite       │
│                                                  │
│  OR receives audio playback:                     │
│    "audio_start" → startPlayback()               │
│    → stops KWS, uninstalls mic I2S              │
│    → sets up speaker I2S                         │
│    Binary chunks → i2s_write() to speaker        │
│    "audio_end" → stopPlayback()                  │
│    → restores mic I2S, restarts KWS              │
└──────────────────────────────────────────────────┘
```

---

## Key Functions (by file)

| File | Key Functions |
|------|--------------|
| `Hardware/MCU.txt` | `setupI2SMIC()`, `capture_samples()`, `audio_inference_callback()`, `runSlice()` (neural net classifier), `startKWS()`/`stopKWS()`, `startStream()`/`stopStream()`, `streamFrame()`, `startPlayback()`/`stopPlayback()`, `webSocketEvent()`, `handleCommandMessage()` |
| `app/vad/vad_detection.py` | `detect_speech()` — WebRTC VAD + 4s silence threshold + 6s no-speech timeout, saves audio, calls `process_audio()` |
| `app/audio/utils.py` | `save_audio()` — raw PCM → WAV file |
| `app/stt/whisper.py` | `transcribe_audio()` — Groq Whisper-large-v3 STT |
| `app/face/face_tools.py` | `FrameBuffer`, `FaceRecognitionBridge.recognize()`, `.register()`, `get_bridge()` |
| `app/face/context.py` | `build_face_context()` — known person → LLM context, `build_unknown_context()` |
| `app/face/person_memory.py` | `PersonMemory.get()`, `.touch()`, `.add_history()`, `.get_summary()` — persistent per-person JSON storage |
| `app/llm/groq.py` | `groq_llm_json()` — Groq LLM call, classifies into command/query/register, returns structured JSON |
| `app/tts/tts_engine.py` | `text_to_speech()` — Kokoro (English, with emotions) or Edge-TTS (Nepali), `play_audio()` |
| `app/communication/websocket.py` | `handle_client()` — receives binary audio from ESP32, `send_websocket_message()`/`broadcast()` — sends text/JSON, `stream_audio()`/`stream_music()` — streams WAV to ESP32 |
| `app/pipeline/process_audio.py` | `process_audio()` — main pipeline: parallel STT + face rec → LLM → TTS → command handling → restart stream |
| `app/orchestration/orchestrator.py` | `Orchestrator` — ties all subsystems: EventBus, StateMachine, FaceMonitor, Greeter, RegistrationManager |
| `app/orchestration/registration.py` | `RegistrationManager` — new user flow: ask name → collect face samples → register |
| `app/orchestration/state.py` | `ConversationStateMachine` — IDLE → FACE_DETECTED → GREETING → CONVERSING → LISTENING → SPEAKING → IDLE |
| `app/orchestration/face_monitor.py` | `FaceMonitor` — background 1s poll loop, publishes face_recognized/face_unknown/face_lost events |

---

## Correction: Where VAD Actually Runs

The user described silence detection as happening on ESP32. **In this codebase, VAD runs on the Pi:**

| Aspect | Pi (`vad_detection.py`) | ESP32 (`MCU.txt`) |
|--------|------------------------|-------------------|
| WebRTC VAD + volume check | ✅ `detect_speech()` | ❌ streams blindly |
| 4-second silence cutoff | ✅ `SILENCE_LIMIT = 200 frames` | ❌ |
| No-speech timeout (6s) | ✅ `NO_SPEECH_LIMIT = 300 frames` | ❌ |
| Saves audio to WAV | ✅ `save_audio()` | ❌ |
| Sends "stop_stream" on end | ✅ | ❌ |
| 10s hard stream timeout | ❌ | ✅ `streamStart > 10000` (fallback) |

The ESP32's only timeout is a 10-second hard cap on streaming duration as a safety net. All intelligent silence detection is on the Pi.

---

## Audio Specs

| Direction | Sample Rate | Bit Depth | Channels | Format |
|-----------|------------|-----------|----------|--------|
| ESP32 → Pi (mic) | 16 kHz | 16-bit | mono | PCM binary frames |
| Pi → ESP32 (TTS) | 24 kHz | 16-bit | mono | WAV streamed in 512-byte chunks |
| Pi → ESP32 (music) | 24 kHz | 16-bit | mono | WAV streamed in 512-byte chunks |
