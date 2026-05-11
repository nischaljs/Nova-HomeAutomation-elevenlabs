🏠 Nova — AI Home Automation Assistant

Nova is a multi-language, voice-enabled AI assistant designed to control home automation, manage media, and respond naturally to user commands. It supports English 🇺🇸 and Nepali 🇳🇵, integrates with IoT devices, and can stream music directly from YouTube using yt-dlp / yt-lm.

🚀 Features
🎤 Voice-based interaction (English & Nepali)
🧠 AI-powered conversational responses
🔊 Emotion-aware Text-to-Speech (TTS)
📡 IoT device control (ESP32-based)
🎵 YouTube music streaming via yt-dlp        
⏯️ Media controls: play, pause, resume, stop
⚡ Real-time automation commands
🧱 System Architecture
User (Voice/Text)
        ↓
   AI Engine (LLM)
        ↓
 JSON Response Parser
        ↓
 ┌───────────────┬───────────────┬───────────────┐
 │               │               │               │
TTS Engine   Command Handler   Music System
 │               │               │
Audio Output   ESP32 / IoT     yt-dlp / yt-lm
🎵 Music System (yt-dlp / yt-lm)

Nova streams audio from YouTube using yt-dlp. The workflow:

User: "Play Arijit Singh songs"
        ↓
AI extracts intent → song query
        ↓
yt-dlp searches YouTube & extracts audio URL
        ↓
Audio played through Pygame / VLC / local audio output
Minimal Python Example
import yt_dlp
import pygame

def get_audio_url(query):
    ydl_opts = {
        'format': 'bestaudio',
        'noplaylist': True,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch:{query}", download=False)
        return info['entries'][0]['url']

url = get_audio_url("Nepali song")
pygame.mixer.init()
pygame.mixer.music.load(url)
pygame.mixer.music.play()

⚠️ Tip: For stable streaming, consider using python-vlc or ffmpeg for direct URL playback.  if you can set it up here a follow up inlk https://youtu.be/cKVJWdeFMa0?si=zQi646kqhHfM1fZ5

🔧 Supported Commands
Play music: "Play music", "Play Arijit Singh songs", "Play Nepali songs"
Pause / resume / stop music
Home automation: "Turn on the lights", "Switch off fan"
Conversational commands: "Hello", "नमस्ते"
🛠️ Tech Stack

Software:

Python 3.9+
AsyncIO (concurrent tasks)
Pygame / VLC (audio playback)
edge-tts (Nepali TTS)
Kokoro ONNX (English TTS)
yt-dlp / yt-lm (YouTube streaming)

Hardware:

ESP32 / ESP32-CAM
Amplifier modules (MAX / TDA7297)
Speaker system
IoT actuators & sensors
📦 Installation
Clone the repository:
git clone https://github.com/your-username/nova-home-automation.git
cd nova-home-automation
Install dependencies:
pip install -r requirements.txt
Install yt-dlp:
pip install yt-dlp
Download Kokoro models:
kokoro-v1.0.fp16-gpu.onnx
voices-v1.0.bin

Place them in the project root.

▶️ Running Nova
python main.py

Example interaction:

User: Turn on the lights
Nova: Lights have been turned on.

User: Play Nepali song
Nova: Playing "Timi Sanga" from YouTube.
🔊 Audio System Setup
ESP32 → I2S / DAC → Amplifier → Speaker
Supports MAX I2S (low power)
TDA7297 / TDA2030 (higher power for bigger speakers)
⚠️ Common Issues
Music not playing: check internet & yt-dlp installation
Nepali TTS not working: check Unicode text & edge-tts config
Distorted sound: check amplifier & speaker wiring
🧠 Future Improvements
Offline speech recognition
Mobile app / dashboard UI
Custom wake-word detection
Streaming Spotify / local music
Context-aware music commands
👨‍💻 Author

Yubraj — AI + Embedded Systems Enthusiast
