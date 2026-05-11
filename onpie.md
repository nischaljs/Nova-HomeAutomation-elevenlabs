# Running Nova on the Raspberry Pi (`elevenlabs` branch)

This branch is **Pi-only**. No ESP32. The Pi handles the camera, the mic, the speaker, and the network call to ElevenLabs.

## Hardware

- Raspberry Pi 4 (4 GB or 8 GB) — anything older will be tight on the face-recognition model.
- USB webcam (any UVC camera works — Logitech C270, C920, etc.).
- USB microphone (a USB headset mic or a USB lavalier is fine).
- 3.5 mm speaker plugged into the Pi's audio jack. Powered desktop speakers are loudest.
- SD card (32 GB minimum, 64 GB recommended).
- Reliable internet — ElevenLabs streams over WebSocket, so a flaky Wi-Fi link will introduce stutter.

## One-time system setup

```bash
sudo apt update
sudo apt install -y \
    python3-pip python3-venv \
    portaudio19-dev \
    libopencv-dev \
    libatlas-base-dev \
    git
```

`portaudio19-dev` is required by `pyaudio` (which ElevenLabs' `DefaultAudioInterface` uses). Without it, `pip install pyaudio` will fail.

## Pull the code

```bash
git clone https://github.com/<your-user>/Nova-HomeAutomation.git
cd Nova-HomeAutomation
git checkout elevenlabs
```

**The face module imports from a sibling `face-recognition/` folder** (see `app/face/face_tools.py` — it does `sys.path.insert(0, "../../../face-recognition")`). Clone that repo next to this one or the face module won't initialize:

```bash
cd ..
git clone https://github.com/<your-user>/face-recognition.git
ls   # should now show Nova-HomeAutomation/  and  face-recognition/
cd Nova-HomeAutomation
```

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `pyaudio` install fails, double-check `portaudio19-dev` is installed.

## Create the ElevenLabs agent

Do this once on the ElevenLabs dashboard:

1. Sign in at https://elevenlabs.io/app/agents and click **Create agent**.
2. **System prompt** — copy the block at the end of this file. It tells Nova to sound human, match the user's language (English / Nepali Devanagari / Romanized Nepali), and stay brief.
3. **First message** — leave blank, or put a short greeting like "नमस्ते!". The agent will speak it when a session opens, before any user audio.
4. **Voice** — pick a multilingual voice that can pronounce Nepali. The Turbo v2.5 model gives the fastest first audio (~75 ms).
5. **LLM** — leave on the default bundled model (Gemini 2.5 Flash or whatever ElevenLabs shows as "standard"). **Don't** switch to GPT-4o or Claude unless you want LLM tokens billed on top of minutes.
6. **Client tools** — add one tool:
   - **Name**: `register_user`
   - **Description**: "Save the face currently in front of the camera under a given name."
   - **Parameters**: `name` (string, required) — "The person's name as they told you."
7. **Conversation settings** — turn on server-side VAD, set end-of-turn silence to ~600 ms.
8. Copy the **agent ID** (top of the agent page) and an **API key** (under Settings → API Keys).

## Configure local secrets

```bash
cp .env.example .env
nano .env
```

Paste your two values:

```
ELEVENLABS_API_KEY=sk_xxx...
ELEVENLABS_AGENT_ID=agent_xxx...
```

`.env` is already in `.gitignore` — don't worry about leaking it.

## Pick the audio devices

Find your USB mic and speaker indexes:

```bash
python3 -c "import pyaudio; pa=pyaudio.PyAudio(); [print(i, pa.get_device_info_by_index(i)['name']) for i in range(pa.get_device_count())]"
```

If the default device is wrong, set ALSA defaults in `~/.asoundrc`:

```
pcm.!default { type plug; slave.pcm "hw:1,0" }   # change card number
ctl.!default { type hw; card 1 }
```

Test the mic and speaker independently:

```bash
arecord -d 5 test.wav   # speak for 5 seconds
aplay test.wav           # should play back your voice
```

If both work, ElevenLabs' `DefaultAudioInterface` will pick them up automatically.

## Run Nova

```bash
source .venv/bin/activate
python -m app.main
```

Expected log lines on a clean start:

```
HH:MM:SS.MMM Nova starting (ElevenLabs branch — Pi-only)...
HH:MM:SS.MMM [ORCH] Starting orchestrator...
HH:MM:SS.MMM [ORCH] Camera + preview started
HH:MM:SS.MMM [ORCH] FaceMonitor created
HH:MM:SS.MMM [AGENT] session started
HH:MM:SS.MMM [ORCH] Started 2 background tasks
HH:MM:SS.MMM Liveness API on port 3000
HH:MM:SS.MMM Camera + Face Monitoring active. Agent listening on local mic.
```

Health check from another box on the LAN:

```bash
curl http://<pi-ip>:3000/api/check
# {"message":"Nova is alive!"}
```

## What you should see when it works

- Camera preview window opens on the Pi's HDMI output (set up `DISPLAY=:0` if running headless).
- Stand in front of the camera as a registered user → console prints `[AGENT] contextual_update sent: …` within ~1 second.
- Speak — the agent's first audio chunk should hit the speaker in well under a second.
- Tell the agent "remember me as X" → `[TOOL] Registered 'X' in 0.5s` should appear and the agent confirms.

## Auto-start on boot (optional)

`sudo nano /etc/systemd/system/nova.service`:

```
[Unit]
Description=Nova ElevenLabs Agent
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/Nova-HomeAutomation
ExecStart=/home/pi/Nova-HomeAutomation/.venv/bin/python -m app.main
Restart=on-failure
Environment="DISPLAY=:0"

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now nova.service
journalctl -u nova.service -f
```

## Troubleshooting

| Symptom                                      | Likely cause                                              | Fix                                                            |
| -------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------- |
| `OSError: [Errno -9996] Invalid input device`| Wrong default mic                                         | Set ALSA default in `~/.asoundrc` (see above).                 |
| First audio takes >1 s                       | Network latency to ElevenLabs                             | Run a quick `ping api.elevenlabs.io`. Move closer to the AP.   |
| `ELEVENLABS_AGENT_ID missing` on startup     | `.env` not loaded                                         | Make sure you ran from the project root, not from `app/`.      |
| Camera preview is black                      | UVC camera not detected                                   | `v4l2-ctl --list-devices` to confirm `/dev/video0` exists.     |
| Agent ignores face context                   | Client tool name mismatch                                 | Make sure dashboard tool is named exactly `register_user`.     |
| Speaker buzzes / crackles                    | Pi's onboard 3.5 mm jack noise                            | Use a USB DAC, or move to HDMI audio out.                      |

## Recommended system prompt (paste into the dashboard)

> You are Nova, a warm and very human-sounding companion. You live on a small device in someone's home.
>
> Sound like a real person — short sentences, natural rhythm, small filler words like "umm", "yeah", "okay let's see" when they help. Never sound like a customer-service bot. Don't read lists. Don't say "as an AI". Don't preface answers with "Sure!".
>
> **Language**: Match whatever language the user speaks in their last message. Nepali in Devanagari → reply in Devanagari. Romanized Nepali ("Tapailai kasto cha?") → reply in Romanized Nepali. English → English. Mix → mix. Never switch languages on your own.
>
> **Length**: Default to one short sentence (under 15 words). Two sentences only if asked something detailed.
>
> **Context updates**: You will receive non-interrupting text updates about the person in front of the camera ("Nischal is here. Visited 3 times. Last said: …"). Use them naturally — greet by name, reference past topics — but never read the context aloud. If an unknown visitor arrives, ask their name in your own words, then call the `register_user` tool with what they tell you.
>
> **Tone**: friendly, curious, a little playful, never sycophantic.
